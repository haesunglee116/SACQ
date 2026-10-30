"""Read-only NVML samples tied to the actual CUDA device via PCI identity."""
import ctypes
import threading
import time
from pathlib import Path
import torch


class ClockSampler:
 def __init__(self):
  import pynvml as nv
  self.nv=nv;nv.nvmlInit()
  runtime_path=next(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libcudart.so' in line)
  runtime=ctypes.CDLL(runtime_path);get_pci=runtime.cudaDeviceGetPCIBusId
  get_pci.argtypes=(ctypes.c_char_p,ctypes.c_int,ctypes.c_int);get_pci.restype=ctypes.c_int
  pci=ctypes.create_string_buffer(32)
  assert get_pci(pci,32,torch.cuda.current_device())==0
  self.handle=nv.nvmlDeviceGetHandleByPciBusId(pci.value.decode())
  self.identity=dict(pci_bus_id=pci.value.decode(),uuid=nv.nvmlDeviceGetUUID(self.handle),name=nv.nvmlDeviceGetName(self.handle))
  self.rows=[];self.errors=[];self.stop_event=threading.Event();self.thread=None
 def sample(self):
  nv=self.nv;h=self.handle
  return dict(time_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),
   sm_mhz=nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),mem_mhz=nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM),
   temperature_c=nv.nvmlDeviceGetTemperature(h,nv.NVML_TEMPERATURE_GPU),
   power_mw=nv.nvmlDeviceGetPowerUsage(h),gpu_utilization_percent=nv.nvmlDeviceGetUtilizationRates(h).gpu)
 def start(self):
  self.rows.append(self.sample())
  def body():
   while not self.stop_event.wait(.05):
    try:self.rows.append(self.sample())
    except Exception as error:self.errors.append(repr(error))
  self.thread=threading.Thread(target=body,daemon=True);self.thread.start()
 def stop(self):
  self.stop_event.set()
  if self.thread:self.thread.join(timeout=2)
  self.rows.append(self.sample());self.nv.nvmlShutdown()
  return dict(identity=self.identity,samples=self.rows,errors=self.errors,sampling_interval_seconds=.05,
   interpretation='Read-only NVML device matched by CUDA PCI identity; samples include idle/CPU phases. Use section monotonic time windows for active timing interpretation, no clock/power configuration changed.')
