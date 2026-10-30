"""Compare existing staged/shared-codebook CUDA kernel against exact decode+GEMM.

Same packed values/A16 operands; changed accumulation rounding is diagnostic.
"""
from cuda_ffn_r3 import packed_cutlass_codebook_linear, load_extension

CONFIGS=[(ctas,splits,bm48) for ctas in (256,384,512) for splits in (4,8) for bm48 in (True,False)]
SELECTED={}
def key(m,x):return (m.out_features,m.in_features,m.index_bits,m.index_window_bytes,x.numel()//x.shape[-1])
def linear(x,m,cfg):
    load_extension().set_ffn_launch_config(*cfg)
    return packed_cutlass_codebook_linear(x,m)
def activate():
    import decoder_tuning
    original=decoder_tuning.linear
    def dispatch(x,m):
        cfg=SELECTED.get(key(m,x))
        if cfg is not None:return linear(x,m,cfg)
        return original(x,m)
    decoder_tuning.linear=dispatch
