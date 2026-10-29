"""Original BF16 SiLU lookup and BF16 multiply fused with unchanged INT8 producer."""
import hashlib,json,types
from pathlib import Path
import torch
import triton
import triton.language as tl
from int8_direct_integer_round_20261028_r1 import round_int8

@triton.jit
def _silu_mul_lookup_activation(X,U,T,Q,S,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B)
    raw=tl.load(X+row*K+col,(row<M)&(col<K),0)
    index=raw.to(tl.uint16,bitcast=True).to(tl.int32)
    silu=tl.load(T+index).to(tl.float32)
    up=tl.load(U+row*K+col,(row<M)&(col<K),0).to(tl.float32)
    value=(silu*up).to(tl.bfloat16).to(tl.float32)
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
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP
        import int8_refined_policy_20261028_r1 as refined
        import int8_torch_epilogue_runtime_r2 as runtime
        state=policy.model._perrow_a8_candidate_state
        tables={};names=set(names)
        for name in sorted(names):
            if not name.endswith('.down_proj'):continue
            parent_name=name[:-len('.down_proj')]
            parent=policy.model.get_submodule(parent_name)
            if type(parent) is not Qwen3VLTextMLP:continue
            assert not parent.training
            assert parent_name+'.gate_proj' in names and parent_name+'.up_proj' in names
            down=parent.down_proj;act=parent.act_fn;gate=parent.gate_proj;up=parent.up_proj;old_down=down.forward
            mode=state['weight_division_modes'][name]
            device=down._a8_perrow_weight_scale.device
            key=(str(device),repr(act))
            if key not in tables:
                inputs=torch.arange(65536,device=device,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                table=act(inputs).contiguous()
                assert table.shape==(65536,) and table.dtype==torch.bfloat16
                tables[key]=table;summary['lookup_bytes']+=table.numel()*table.element_size()
                summary['tables'].append({'device':str(device),'approximate':repr(act),'bf16_input_patterns':65536,'table_sha256':hashlib.sha256(table.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()})
                del inputs
            table=tables[key]
            def forward(this,x,*args,_act=act,_gate=gate,_up=up,_down=down,_old=old_down,_mode=mode,_table=table,**kwargs):
                assert not args and not kwargs and not this.training
                raw=_gate(x)
                up_output=_up(x)
                assert up_output.shape==raw.shape and up_output.dtype==torch.bfloat16
                assert raw.dtype==torch.bfloat16
                a=raw.reshape(-1,_down.in_features).contiguous();m,k=a.shape
                assert (m,k) in warps
                mp=max(32,triton.cdiv(m,16)*16)
                aq=torch.empty((mp,k),device=a.device,dtype=torch.int8)
                sa=torch.empty((mp,1),device=a.device,dtype=torch.float32)
                _silu_mul_lookup_activation[(mp,)](a,up_output.reshape(-1,k).contiguous(),_table,aq,sa,m,k,triton.next_power_of_2(k),num_warps=warps[(m,k)],enable_fp_fusion=False)
                wq=refined.decoder.decode(_down,_mode)
                y=runtime.linear(aq,wq,sa,_down._a8_perrow_weight_scale,_down.bias,True,actual_rows=m)
                y=y.reshape(*raw.shape[:-1],_down.out_features)
                if state['validation_enabled']:
                    original_gelu=_act(raw)*up_output
                    expected_q,expected_s=refined.original.quantize_rows(original_gelu.reshape(-1,k),'int8_symmetric')
                    assert torch.equal(aq[:m],expected_q),'SwiGLULookupActualINT8Bytes'
                    assert torch.equal(sa[:m].view(torch.int32),expected_s.view(torch.int32)),'SwiGLULookupActualScaleBytes'
                    expected=_old(original_gelu)
                    assert torch.equal(y.view(torch.int16),expected.view(torch.int16)),'SwiGLULookupDownOutputBytes'
                    summary['actual_operand_checks']+=1;summary['actual_output_checks']+=1
                return y
            parent.forward=types.MethodType(forward,parent)
            summary['modules'].append(name)
        assert summary['modules'],'NoCompatibleTextMLP'
        result['exact_swiglu_lookup']=summary
        return result
    fast.install=install
    return summary
