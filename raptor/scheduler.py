import numpy as np
import torch

class CosineScheduler:
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.optimizer = optimizer
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = torch.tensor(np.concatenate((warmup_schedule, schedule))).float()
        self.iter = 0

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        return self.final_value if it >= self.total_iters else self.schedule[it]

    def step(self):
        self.iter += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self[self.iter]