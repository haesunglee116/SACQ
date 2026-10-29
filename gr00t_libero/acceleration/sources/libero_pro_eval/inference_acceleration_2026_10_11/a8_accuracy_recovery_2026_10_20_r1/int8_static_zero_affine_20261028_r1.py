"""Fixed-checkpoint zero-shift specialization; installer must fail on nonzero shift."""
import triton
import triton.language as tl
@triton.jit
def _affine_bf16x4(word,scale,shift):
 lo=word.to(tl.uint32);hi=(word>>32).to(tl.uint32)
 ss=scale.to(tl.uint16,bitcast=True).to(tl.uint32)*0x10001
 a,b=tl.inline_asm_elementwise(
  '{ .reg .b32 nz; mov.b32 nz, 0x80008000; fma.rn.bf16x2 $0, $2, $4, nz; fma.rn.bf16x2 $1, $3, $4, nz; }',
  constraints='=r,=r,r,r,r',args=[lo,hi,ss],dtype=(tl.uint32,tl.uint32),is_pure=True,pack=1)
 return a.to(tl.uint64)|(b.to(tl.uint64)<<32)
