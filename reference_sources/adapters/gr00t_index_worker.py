"""Frozen quantized-model full-H; original SACQ vector correction numerical core."""
import fcntl,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from oft_index_worker import load,save,sha
R=Path(__file__).resolve().parent;W=R.parent;B=W/'libero_pro_eval'
def main(path):
    j=load(path);out=Path(j['out']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for k in list(os.environ):
        if k.startswith(('VLM_','DIT_','SACQ_','ACTION_HEAD_')):os.environ.pop(k)
    sys.path[:0]=[str(B),str(B/'Isaac-GR00T-N1.7'),str(W/'simplerenv_n17_feasibility_2026_08_27')]
    import torch,transformers
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    for cls in (transformers.AutoProcessor,transformers.Qwen3VLProcessor,transformers.Qwen3VLForConditionalGeneration):
        old=cls.from_pretrained
        def local(cls,name,*args,_old=old,**kwargs):return _old(j['processor'] if name=='nvidia/Cosmos-Reason2-2B' else name,*args,**kwargs)
        cls.from_pretrained=classmethod(local)
    sp=importlib.util.spec_from_file_location('workshop_trainer',j['trainer']);ft=importlib.util.module_from_spec(sp);sp.loader.exec_module(ft)
    from gr00t.policy.server_client import MsgSerializer
    from measure_hotspot_protection_proxy import collate_request,move_to_device
    import sacq_compensation_stats_r1_2026_09_10 as stats
    import sacq_vector_compensation_core_r1_2026_09_10 as core
    from gr00t_fixed_model import install
    policy=ft.Gr00tPolicy(ft.EmbodimentTag.resolve('LIBERO_PANDA'),j['model'],device='cuda:0',strict=True)
    originals={r['layer']:policy.model.get_submodule(r['layer']).weight.detach().cpu().clone() for r in j['targets']}
    install(policy,ft,j['allocation'],j['artifacts'],j['gptq'])
    moments={};handles=[];outputs=[];started=time.time()
    for r in j['targets']:
        name=r['layer'];receipt=out/(name+'.json')
        if receipt.exists():
            rr=load(receipt);assert rr['job_sha256']==sha(path) and sha(rr['path'])==rr['sha256'];outputs.append(rr);continue
        m=policy.model.get_submodule(name);moments[name]=stats.FullMoment(m.in_features,'cuda',2048)
        def hook(m,args,result,name=name):moments[name].add(args[0])
        handles.append(m.register_forward_hook(hook))
    versions={n:(id(p),p._version) for n,p in list(policy.model.named_parameters())+list(policy.model.named_buffers())}
    try:
        with torch.no_grad():
            for i,row in enumerate(j['records'] if moments else []):
                p=MsgSerializer.from_bytes(Path(row['path']).read_bytes());bb,ai=policy.model.prepare_input(collate_request(policy,p['observation']))
                for m in moments.values():m.begin_request()
                action=ft.run_action_grad(policy,move_to_device(bb,torch.device('cuda'),torch.bfloat16),move_to_device(ai,torch.device('cuda'),torch.bfloat16),seed=7000+row['position'])
                assert torch.isfinite(action).all()
                for r in j['targets']:
                    if r['layer'] in moments:
                        m=moments[r['layer']];assert m.request_calls==(1 if r['scope']=='vlm' else policy.model.action_head.num_inference_timesteps) and m.request_rows>0
                if i==0 or (i+1)%16==0:save(out/'STATE.json',dict(stage='COLLECTING',requests=i+1,total=576,elapsed=time.time()-started));print('REQUEST',i+1,576,flush=True)
    finally:
        for h in handles:h.remove()
    assert versions=={n:(id(p),p._version) for n,p in list(policy.model.named_parameters())+list(policy.model.named_buffers())}
    for r in j['targets']:
        name=r['layer']
        if name not in moments:continue
        m=moments.pop(name);H=m.total.div_(m.rows);p=torch.load(r['path'],map_location='cpu',weights_only=False)
        save(out/'STATE.json',dict(stage='CORRECTING',layer=name,finished=len(outputs),total=len(j['targets'])))
        q,report=core.compensate_packed(p['packed'],originals.pop(name).to('cuda'),H,checkpoint_dir=None,
            provenance=dict(job_sha256=sha(path),source=r),block_columns=128,damping=.1)
        core.assert_only_indices_changed(p['packed'],q)
        dest=out/'bank'/(hashlib.sha256(name.encode()).hexdigest()[:24]+'.pt');stats.atomic_torch(dest,{'packed':q})
        rr=dict(layer=name,path=str(dest),sha256=sha(dest),job_sha256=sha(path),indices_only_verified=True,report=report,hook_calls=m.calls,token_rows=m.rows)
        save(out/(name+'.json'),rr);outputs.append(rr);del m,H,q,p;torch.cuda.empty_cache();print('LAYER_COMPLETE',name,flush=True)
    save(out/'COMPLETE.json',dict(status='COMPLETE_DIAGNOSTIC',requests=576,outputs=outputs,job_sha256=sha(path),held_out=False))
if __name__=='__main__':main(sys.argv[1])
