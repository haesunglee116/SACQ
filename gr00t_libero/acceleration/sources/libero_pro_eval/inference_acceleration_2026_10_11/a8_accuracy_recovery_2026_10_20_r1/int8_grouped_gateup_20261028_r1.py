"""Same-input gate/up projection GEMMs in one launch; zero third tile extent."""
import types
import torch
import triton
import triton.language as tl

@triton.jit
def _int8_grouped_gateup(A,SA,W0,W1,W2,S0,S1,S2,B0,B1,B2,Y0,Y1,Y2,
                      M:tl.constexpr,N0:tl.constexpr,N1:tl.constexpr,N2:tl.constexpr,K:tl.constexpr,
                      BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(1)
    T0:tl.constexpr=triton.cdiv(N0,BN)
    T1:tl.constexpr=triton.cdiv(N1,BN)
    first=tile<T0;second=tile<T0+T1
    W=tl.where(first,W0,tl.where(second,W1,W2))
    SW=tl.where(first,S0,tl.where(second,S1,S2))
    B=tl.where(first,B0,tl.where(second,B1,B2))
    Y=tl.where(first,Y0,tl.where(second,Y1,Y2))
    N=tl.where(first,N0,tl.where(second,N1,N2))
    nt=tl.where(first,tile,tl.where(second,tile-T0,tile-T0-T1))
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=nt*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.int32)
    for block in range(triton.cdiv(K,BK)):
        kk=block*BK+rk
        a=tl.load(A+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0)
        w=tl.load(W+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0)
        acc=tl.dot(a,w,acc,out_dtype=tl.int32)
    sw=tl.load(SW+rn,rn<N,1.0)
    sa=tl.load(SA+rm,rm<M,1.0)
    scaled=acc.to(tl.float32)*sw[None,:]
    if BIAS:
        bias=tl.load(B+rn,rn<N,0).to(tl.float32)
        value=tl.fma(sa[:,None],scaled,bias[None,:])
    else:value=sa[:,None]*scaled
    tl.store(Y+rm[:,None]*N+rn[None,:],value.to(tl.bfloat16),(rm[:,None]<M)&(rn[None,:]<N))

def compute(x,modules,modes):
    import int8_refined_policy_20261028_r1 as refined
    assert len(modules)==len(modes)==2
    k=modules[0].in_features;a=x.reshape(-1,k).contiguous();m=a.shape[0]
    assert m<=160 and k*127*127<=torch.iinfo(torch.int32).max
    aq,sa=refined.producer.activation(a,0)
    weights=[refined.decoder.decode(mod,mode) for mod,mode in zip(modules,modes)]
    ns=[mod.out_features for mod in modules]
    outputs=[torch.empty((m,n),dtype=torch.bfloat16,device=x.device) for n in ns]
    biases=[mod.bias if mod.bias is not None else outputs[i] for i,mod in enumerate(modules)]
    _int8_grouped_gateup[(triton.cdiv(m,64),sum(triton.cdiv(n,64) for n in ns))](
        aq,sa,weights[0],weights[1],weights[1],
        modules[0]._a8_perrow_weight_scale,modules[1]._a8_perrow_weight_scale,modules[1]._a8_perrow_weight_scale,
        biases[0],biases[1],biases[1],outputs[0],outputs[1],outputs[1],
        m,ns[0],ns[1],0,k,modules[0].bias is not None,64,64,128,num_warps=4,num_stages=3,enable_fp_fusion=False)
    return [y.reshape(*x.shape[:-1],n) for y,n in zip(outputs,ns)]

def enable():
    import perrow_int8_fast_policy_r3 as fast
    previous=fast.install
    summary={'groups':[],'validated_outputs':0,'calls':0}
    def install(policy,names,contract):
        result=previous(policy,names,contract)
        state=policy.model._perrow_a8_candidate_state
        names=set(names)
        for parent_name in sorted({n.rsplit('.',1)[0] for n in names}):
            group=next((g for g in [('gate_proj','up_proj')] if all(parent_name+'.'+n in names for n in g)),None)
            if group is None:continue
            full=[parent_name+'.'+n for n in group]
            modules=[policy.model.get_submodule(n) for n in full]
            # Cross-attention differs in input features and is excluded.
            if len({mod.in_features for mod in modules})!=1:continue
            if len({mod.bias is not None for mod in modules})!=1:continue
            parent=policy.model.get_submodule(parent_name)
            frames=[]
            def enter(this,args,_frames=frames):_frames.append({})
            def leave(this,args,output,_frames=frames):_frames.pop()
            parent.register_forward_pre_hook(enter)
            parent.register_forward_hook(leave,always_call=True)
            modes=[state['weight_division_modes'][n] for n in full]
            for i,mod in enumerate(modules):
                old=mod.forward
                def forward(this,x,_i=i,_old=old,_frames=frames,_modules=modules,_modes=modes):
                    assert _frames,'GroupedQKVOutsideParent'
                    frame=_frames[-1]
                    key=(x.data_ptr(),tuple(x.shape),tuple(x.stride()))
                    if 'outputs' not in frame:
                        frame.update(source=x,key=key,outputs=compute(x,_modules,_modes))
                    assert frame['key']==key,'GroupedQKVInputsDiffer'
                    y=frame['outputs'][_i]
                    if state['validation_enabled']:
                        expected=_old(x)
                        assert torch.equal(y.view(torch.int16),expected.view(torch.int16)),'GroupedQKVOutputBytes'
                        summary['validated_outputs']+=1
                    summary['calls']+=1
                    return y
                mod.forward=types.MethodType(forward,mod)
            summary['groups'].append(full)
        assert summary['groups'],'NoGroupedQKVParents'
        result['grouped_gateup']=summary
        return result
    fast.install=install
    return summary
