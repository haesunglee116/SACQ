def require(condition, message):
    if not condition:
        raise ValueError(message)


class FullMoment:
    """FP32 FULL uncentered sum; row chunking limits scratch, never drops rows."""

    def __init__(self, width, device, chunk_rows):
        import torch
        require(width > 0 and chunk_rows > 0, 'Invalid accumulator geometry')
        self.total = torch.zeros((width, width), dtype=torch.float32, device=device)
        self.width, self.chunk_rows = width, chunk_rows
        self.rows = self.calls = self.request_rows = self.request_calls = 0

    def begin_request(self):
        self.request_rows = self.request_calls = 0

    def add(self, x):
        import torch
        require(isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[-1] == self.width,
                'Hook input shape differs from Linear input width')
        flat = x.detach().reshape(-1, self.width)
        require(flat.shape[0] > 0 and x.is_floating_point(), 'Empty/nonfloating hook input')
        for start in range(0, flat.shape[0], self.chunk_rows):
            part = flat[start:start+self.chunk_rows].to(device=self.total.device, dtype=torch.float32)
            require(bool(torch.isfinite(part).all()), 'Nonfinite BF16 background input')
            self.total.addmm_(part.T, part)
        n = int(flat.shape[0])
        self.rows += n
        self.calls += 1
        self.request_rows += n
        self.request_calls += 1

    def state(self):
        return dict(total=self.total.cpu(), rows=self.rows, calls=self.calls)

    def restore(self, state):
        import torch
        require(state['total'].shape == self.total.shape and state['total'].dtype == torch.float32,
                'Checkpoint matrix geometry/dtype mismatch')
        require(bool(torch.isfinite(state['total']).all()), 'Nonfinite checkpoint')
        self.total.copy_(state['total'])
        self.rows, self.calls = int(state['rows']), int(state['calls'])

