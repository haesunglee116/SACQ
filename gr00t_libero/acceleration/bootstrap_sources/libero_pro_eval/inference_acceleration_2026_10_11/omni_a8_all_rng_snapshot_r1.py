"""Explicit state identities for every RNG named by the approved native gate."""
import hashlib,pickle,random
import numpy as np
import torch

def _sha(data):return hashlib.sha256(data).hexdigest()
def snapshot():
 return dict(torch_CPU=_sha(torch.get_rng_state().numpy().tobytes()),
             numpy=_sha(pickle.dumps(np.random.get_state(),protocol=4)),
             python=_sha(pickle.dumps(random.getstate(),protocol=4)),
             torch_CUDA=_sha(torch.cuda.get_rng_state().cpu().numpy().tobytes()))
