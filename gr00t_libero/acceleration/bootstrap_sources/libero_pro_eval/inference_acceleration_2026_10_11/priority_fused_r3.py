"""New grouped, pipelined large-M tile trials. No retained decoded weights.

Language uses d4 codewords; Vision loads each packed byte once and interleaves
its two dictionary values. Stock representations and A16 operands are retained.
"""
import torch
import triton as tr
import triton.language as tl

SELECTED={}
COMPILED={}
CONFIGS=[(32,32,64,4,1),(32,64,64,4,1),(64,64,64,8,1),
         (64,64,64,8,2),(64,128,64,8,1),(128,64,64,8,1)]

@tr.jit
def _tile(X,P,C,S,Z,B,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
          G:tl.constexpr,BITS:tl.constexpr,WORDS:tl.constexpr,DICT:tl.constexpr,
          HAS_B:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,ST:tl.constexpr):
    # Group neighbouring M tiles to reuse the same packed B tile in L2.
    pid=tl.program_id(0);nm=tr.cdiv(M,BM);nn=tr.cdiv(N,BN)
    group=pid//(8*nn);first=group*8;actual=tl.minimum(nm-first,8)
    pm=first+(pid%(8*nn))%actual;pn=(pid%(8*nn))//actual
    mi=pm*BM+tl.arange(0,BM);ni=pn*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in tl.range(0,tr.cdiv(K,BK),num_stages=ST):
        kk=start*BK+tl.arange(0,BK)
        a=tl.load(X+mi[:,None]*K+kk[None,:],(mi[:,None]<M)&(kk[None,:]<K),0)
        if DICT:
            kp=start*(BK//2)+tl.arange(0,BK//2)
            wi=ni[:,None]*K+kp[None,:]*2
            mask=(ni[:,None]<N)&(kp[None,:]<K//2)
            byte=tl.load(P+wi//2,mask,0).to(tl.int32)
            offset=(wi//G)*16
            lo=tl.load(C+offset+(byte&15),mask,0)
            hi=tl.load(C+offset+(byte>>4),mask,0)
            w=tl.interleave(lo,hi)
        else:
            vi=start*(BK//4)+tl.arange(0,BK//4)
            mask=(ni[:,None]<N)&(vi[None,:]<K//4)
            bit=(ni[:,None]*(K//4)+vi[None,:])*BITS
            word=bit//32;off=bit%32
            low=tl.load(P+word,mask&(word<WORDS),0)
            high=tl.load(P+word+1,mask&(off+BITS>32)&(word+1<WORDS),0)
            idx=((low>>off)|(high<<((32-off)%32)))&((1<<BITS)-1)
            q=tl.load(C+idx[:,:,None]*4+tl.arange(0,4)[None,None,:],mask[:,:,None],0)
            scale=tl.load(S+ni[:,None]*(K//G)+vi[None,:]//(G//4),mask,0)
            shift=tl.load(Z)
            w=((q*scale[:,:,None]).to(tl.bfloat16)+shift).to(tl.bfloat16)
            w=tl.reshape(w,(BN,BK))
        acc=tl.dot(a,tl.trans(w),acc)
    if HAS_B:acc+=tl.load(B+ni,ni<N,0)[None,:].to(tl.float32)
    tl.store(Y+mi[:,None]*N+ni[None,:],acc,(mi[:,None]<M)&(ni[None,:]<N))


def key(m,x,region):
    return (region,m.out_features,m.in_features,x.numel()//x.shape[-1],
            getattr(m,'index_bits',getattr(m,'bits',None)),m.group_size,m.bias is not None)


def linear(x,m,cfg):
    assert x.dtype==torch.bfloat16 and x.is_cuda
    xx=x.reshape(-1,m.in_features).contiguous();M,K=xx.shape;N=m.out_features
    bm,bn,bk,nw,st=cfg;isdict=hasattr(m,'packed')
    if isdict:
        assert m.bits==4 and m.group_size%2==0
        p=m.packed;cb=m.levels;scale=shift=cb;bits=4;words=0
    else:
        assert m.vector_dim==4 and K%m.group_size==0 and m.group_size%4==0
        count=N*K//4
        assert count*m.index_bits%32==0
        p=m.packed_indices[:m.packed_indices.numel()//4*4].view(torch.uint32)
        cb=m.codebook;scale=m.scales;shift=m.shift;bits=m.index_bits;words=p.numel()
    y=torch.empty((M,N),device=x.device,dtype=x.dtype)
    compiled=_tile[(tr.cdiv(M,bm)*tr.cdiv(N,bn),)](xx,p,cb,scale,shift,m.bias if m.bias is not None else y,y,
        M,N,K,m.group_size,bits,words,isdict,m.bias is not None,bm,bn,bk,st,
        num_warps=nw,num_stages=st,enable_fp_fusion=False)
    COMPILED[(isdict,N,K,M,bits,tuple(cfg))]=dict(n_regs=compiled.n_regs,n_spills=compiled.n_spills,shared_bytes=compiled.metadata.shared,num_warps=compiled.metadata.num_warps)
    return y.reshape(*x.shape[:-1],N)


def activate(model):
    import decoder_tuning as dec
    old=dec.linear
    regions={id(m):('language' if '.language_model.' in name else 'action') for name,m in model.named_modules() if hasattr(m,'packed_indices')}
    def dispatch(x,m):
        cfg=SELECTED.get(key(m,x,regions.get(id(m),'action')))
        if cfg is None:return old(x,m)
        return linear(x,m,cfg[1:]) if cfg[0]=='triton' else cuda_linear(x,m,cfg[1:])
    dec.linear=dispatch
    # exact_dictionary_decode.install runs before Backbone graph capture.
    import exact_dictionary_decode as d
    original=d.install
    def install(model):
        receipt=original(model)
        for name,m in model.named_modules():
            if '.visual.' not in name or not hasattr(m,'packed'):continue
            oldforward=m.forward
            def forward(x,m=m,oldforward=oldforward):
                cfg=SELECTED.get(key(m,x,'vision'))
                return linear(x,m,cfg[1:]) if cfg is not None else oldforward(x)
            m.forward=forward
        return receipt
    d.install=install


def cuda_linear(x,m,cfg):
    from fused_action_candidates_r1 import linear as cuda
    return cuda(x,m,tuple(cfg))
