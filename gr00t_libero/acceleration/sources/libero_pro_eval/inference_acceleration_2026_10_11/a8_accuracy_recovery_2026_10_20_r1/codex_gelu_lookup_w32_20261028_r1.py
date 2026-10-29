"""Original runtime BF16 GELU lookup fused with unchanged per-row INT8 producer."""
import hashlib,json,types
from pathlib import Path
import torch
import triton
import triton.language as tl
from int8_direct_integer_round_20261028_r1 import round_int8

@triton.jit
def _gelu_lookup_activation(X,T,Q,S,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B)
    raw=tl.load(X+row*K+col,(row<M)&(col<K),0)
    index=raw.to(tl.uint16,bitcast=True).to(tl.int32)
    value=tl.load(T+index).to(tl.float32)
    maximum=tl.max(tl.abs(value),0)
    inverse=tl.full((),1.0/127.0,tl.float32)
    scale=tl.where(maximum>0,maximum*inverse,1.0)
    encoded=round_int8(tl.minimum(tl.maximum(tl.div_rn(value,scale),-127.0),127.0))
    tl.store(Q+row*K+col,encoded,col<K);tl.store(S+row,scale)

def enable(selection_ref):
    raw=Path(selection_ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==selection_ref['sha256']
    warps={tuple(r['shape']):r['selected_warps'] for r in json.loads(raw)}
    import perrow_int8_fast_policy_r3 as fast
    previous=fast.install
    summary={'modules':[],'actual_operand_checks':0,'actual_output_checks':0,'lookup_bytes':0,'tables':[]}
    def install(policy,names,contract):
        result=previous(policy,names,contract)
        from diffusers.models.attention import FeedForward
        from diffusers.models.activations import GELU
        import int8_refined_policy_20261028_r1 as refined
        import int8_torch_epilogue_runtime_r2 as runtime
        state=policy.model._perrow_a8_candidate_state
        tables={};names=set(names)
        for name in sorted(names):
            if not name.endswith('.net.2'):continue
            parent_name=name[:-len('.net.2')]
            parent=policy.model.get_submodule(parent_name)
            if type(parent) is not FeedForward or type(parent.net[0]) is not GELU:continue
            assert not parent.training and len(parent.net) in (3,4)
            if len(parent.net)==4:assert type(parent.net[3]) is torch.nn.Dropout and not parent.net[3].training
            assert type(parent.net[1]) is torch.nn.Dropout and not parent.net[1].training
            assert parent_name+'.net.0.proj' in names
            down=parent.net[2];act=parent.net[0];old_down=down.forward
            mode=state['weight_division_modes'][name]
            device=down._a8_perrow_weight_scale.device
            key=(str(device),act.approximate)
            if key not in tables:
                inputs=torch.arange(65536,device=device,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                table=act.gelu(inputs).contiguous()
                assert table.shape==(65536,) and table.dtype==torch.bfloat16
                tables[key]=table;summary['lookup_bytes']+=table.numel()*table.element_size()
                summary['tables'].append({'device':str(device),'approximate':act.approximate,'bf16_input_patterns':65536,'table_sha256':hashlib.sha256(table.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()})
                del inputs
            table=tables[key]
            def forward(this,x,*args,_act=act,_down=down,_old=old_down,_mode=mode,_table=table,**kwargs):
                assert not args and not kwargs and not this.training
                raw=_act.proj(x)
                assert raw.dtype==torch.bfloat16
                a=raw.reshape(-1,_down.in_features).contiguous();m,k=a.shape
                assert (m,k) in warps
                mp=max(32,triton.cdiv(m,16)*16)
                aq=torch.empty((mp,k),device=a.device,dtype=torch.int8)
                sa=torch.empty((mp,1),device=a.device,dtype=torch.float32)
                _gelu_lookup_activation[(mp,)](a,_table,aq,sa,m,k,triton.next_power_of_2(k),num_warps=32,enable_fp_fusion=False)
                wq=refined.decoder.decode(_down,_mode)
                y=runtime.linear(aq,wq,sa,_down._a8_perrow_weight_scale,_down.bias,True,actual_rows=m)
                y=y.reshape(*raw.shape[:-1],_down.out_features)
                if state['validation_enabled']:
                    original_gelu=_act.gelu(raw)
                    expected_q,expected_s=refined.original.quantize_rows(original_gelu.reshape(-1,k),'int8_symmetric')
                    assert torch.equal(aq[:m],expected_q),'GELULookupActualINT8Bytes'
                    assert torch.equal(sa[:m].view(torch.int32),expected_s.view(torch.int32)),'GELULookupActualScaleBytes'
                    expected=_old(original_gelu)
                    assert torch.equal(y.view(torch.int16),expected.view(torch.int16)),'GELULookupDownOutputBytes'
                    summary['actual_operand_checks']+=1;summary['actual_output_checks']+=1
                return this.net[3](y) if len(this.net)==4 else y
            parent.forward=types.MethodType(forward,parent)
            summary['modules'].append(name)
        assert summary['modules'],'NoCompatibleGELUFeedForward'
        result['exact_gelu_lookup']=summary
        return result
    fast.install=install
    return summary
