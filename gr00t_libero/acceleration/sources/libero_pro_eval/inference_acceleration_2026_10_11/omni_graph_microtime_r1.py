"""Event nodes inside micrographs, using the existing dedicated model stream."""
import ctypes
import statistics
from pathlib import Path
import torch


def measure(fn,contract):
    path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if '/libcudart.so' in line)
    runtime=ctypes.CDLL(path)
    record=runtime.cudaEventRecordWithFlags
    record.argtypes=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint)
    record.restype=ctypes.c_int
    stream=torch.cuda.current_stream()
    assert stream.cuda_stream!=0,'Use the existing dedicated model stream'
    for _ in range(contract['warmup']):fn()
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();end.record();end.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        assert record(start.cuda_event,stream.cuda_stream,1)==0
        value=fn()
        assert record(end.cuda_event,stream.cuda_stream,1)==0
    samples=[]
    for _ in range(contract['num_iterations']):
        graph.replay();end.synchronize()
        samples.append(start.elapsed_time(end)*1000)
    output=value.clone()
    del graph,value
    return dict(median_us=statistics.median(samples),samples_us=samples,
        event_placement='external CUDA timing-event nodes inside the same graph',
        existing_model_stream_reused=True,cuda_runtime_library=path),output
