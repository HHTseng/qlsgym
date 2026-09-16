"""Replay buffer of qMDP transitions (arXiv:2608.03702 App. C.1)."""

from __future__ import annotations


class Replay:
    """Pre-allocated, device-resident ring buffer of qMDP transitions."""

    def __init__(self, capacity: int, n_states: int, device, torch):
        self.torch = torch
        self.capacity = int(capacity)
        self.device = device
        f = dict(dtype=torch.float32, device=device)
        self.s = torch.zeros(self.capacity, n_states, **f)
        self.s0 = torch.zeros(self.capacity, n_states, **f)
        self.s1 = torch.zeros(self.capacity, n_states, **f)
        self.a = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.pi0 = torch.zeros(self.capacity, **f)
        self.pi1 = torch.zeros(self.capacity, **f)
        self.r0 = torch.zeros(self.capacity, **f)
        self.r1 = torch.zeros(self.capacity, **f)
        self.d0 = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        self.d1 = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        self.n = 0
        self.pos = 0

    def push(self, s, a, s0, s1, pi0, pi1, r0, r1, d0, d1) -> None:
        b = s.shape[0]
        idx = (self.pos + self.torch.arange(b, device=self.device)) % self.capacity
        self.s[idx] = s.to(self.torch.float32)
        self.s0[idx] = s0.to(self.torch.float32)
        self.s1[idx] = s1.to(self.torch.float32)
        self.a[idx] = a
        self.pi0[idx] = pi0.to(self.torch.float32)
        self.pi1[idx] = pi1.to(self.torch.float32)
        self.r0[idx] = r0.to(self.torch.float32)
        self.r1[idx] = r1.to(self.torch.float32)
        self.d0[idx] = d0
        self.d1[idx] = d1
        self.pos = int((self.pos + b) % self.capacity)
        self.n = int(min(self.n + b, self.capacity))

    def sample(self, batch: int, generator=None):
        i = self.torch.randint(0, self.n, (batch,), device=self.device, generator=generator)
        return (self.s[i], self.a[i], self.s0[i], self.s1[i],
                self.pi0[i], self.pi1[i], self.r0[i], self.r1[i],
                self.d0[i], self.d1[i])

    def __len__(self) -> int:
        return self.n
