"""Original workshop calibration math with corrected bank and frozen GPTQ weights.

An explicit trainer callback installs the supplied frozen background weights
after the original strict quantizer/count checks and before any optimizer step.
No optimizer, forward, loss, ordering or RNG implementation is replaced.
"""
import importlib.util, os, sys, shutil
from pathlib import Path
from io_utils import load,save,sha
R=Path(__file__).resolve().parent; W=R.parent; B=W/'libero_pro_eval'
def main(path):
    j=load(path)
    from release_configuration import calibration_argv
    trainer_argv=calibration_argv(j)
    from release_cache_identity import calibration_paths
    qa_path,batch_path=calibration_paths(j,sha(path))
    for k in list(os.environ):
        if k.startswith(('VLM_','DIT_','SACQ_','ACTION_HEAD_')):os.environ.pop(k)
    sys.path[:0]=[str(B),str(B/'Isaac-GR00T-N1.7')]
    import torch, transformers
    torch.set_num_threads(4)
    for cls in (transformers.AutoProcessor,transformers.Qwen3VLProcessor,transformers.Qwen3VLForConditionalGeneration):
        old=cls.from_pretrained
        def local(cls,name,*args,_old=old,**kwargs):
            return _old(j['processor'] if name=='nvidia/Cosmos-Reason2-2B' else name,*args,**kwargs)
        cls.from_pretrained=classmethod(local)
    def background(policy,trainables,device):
        names={n for n,_ in trainables};assert names=={x['layer'] for x in j['artifacts']}
        assert len(names)==j['sacq_count'] and len(j['gptq'])==j['gptq_count']
        for r in j['gptq']:
            name=r['layer'];assert name not in names
            p=torch.load(r['path'],map_location='cpu',weights_only=False)
            assert p['layer']==name and p['bits']==r['bits']
            old=policy.model.get_submodule(name);bias=getattr(old,'bias',None);w=p['weight']
            new=torch.nn.Linear(w.shape[1],w.shape[0],bias=bias is not None,device=device,dtype=torch.bfloat16)
            with torch.no_grad():
                new.weight.copy_(w)
                if bias is not None:new.bias.copy_(bias)
            new.requires_grad_(False);trainer.replace_module(policy.model,name,new)
        assert {n for n,p in policy.model.named_parameters() if p.requires_grad}=={n+'.'+key for n,q in trainables for key,p in q.named_parameters() if p.requires_grad}
        print('FROZEN_GPTQ_INSTALLED',j['gptq_count'],'SACQ_TRAINABLE',j['sacq_count'],flush=True)
    trainer_spec=importlib.util.spec_from_file_location('train_centroid_ft_2026_08_05',j['trainer'])
    trainer=importlib.util.module_from_spec(trainer_spec)
    sys.modules[trainer_spec.name]=trainer
    trainer_spec.loader.exec_module(trainer)
    original_save=trainer.atomic_torch_save
    def durable_save(payload,path):
        original_save(payload,path)
        target=Path(j['out'])/Path(path).name
        if target.resolve()==Path(path).resolve():
            return
        temp=target.with_suffix(target.suffix+'.tmp')
        with Path(path).open('rb') as src,temp.open('wb') as dest:
            shutil.copyfileobj(src,dest,8*1024*1024);dest.flush();os.fsync(dest.fileno())
        temp.replace(target)
    trainer.atomic_torch_save=durable_save
    trainer.local_ckpt=lambda suite:j['model']
    trainer.collect_requests=lambda *a,**kw:[Path(r['path']) for r in j['all_records']]
    trainer.QA=qa_path
    import artifact_registry
    artifacts={r['layer']:r for r in j['artifacts']}
    def fitted(kind,name,k,suite='libero_spatial',init='aware',verify=True,exact=False):
        r=artifacts[name];assert r['scope']==kind and int(r['k'])==k and init=='aware' and suite==j['suite']
        return Path(r['path'])
    artifact_registry.find_packed=fitted
    spec=importlib.util.spec_from_file_location('transfer_cached_batches',j['cached_launcher'])
    cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
    os.environ['CENTROID_BATCH_CACHE_ROOT']=str(batch_path)
    sys.argv=trainer_argv
    cache.main(install_background=background,output_dir=j['local_output'])
    if int(os.environ.get('RANK','0'))==0:
        out=Path(j['local_output']);latest=torch.load(out/'latest_state.pt',map_location='cpu',mmap=True,weights_only=False)
        if latest['__meta__']['steps_done']!=j['training']['expected_updates']:
            raise RuntimeError('Calibration completion update count differs from explicit job')
        outputs={}
        for key,name in (('vlm', 'vlm_ft_overlay.pt'),
                         ('dit', f"dit_ft_overlay_{j['suite']}.pt"),
                         ('best_state', 'codebook_state.pt')):
            artifact=out/name
            outputs[key]=dict(path=str(artifact.resolve()),sha256=sha(artifact))
        save(out/'TRAINING_COMPLETE.json',dict(status='COMPLETE_DIAGNOSTIC',
            updates=latest['__meta__']['steps_done'],job_sha256=sha(path),outputs=outputs))
if __name__=='__main__':main(sys.argv[1])

