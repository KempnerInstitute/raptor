import pytest
import torch

from scheduler import CosineScheduler


def test_cosine_scheduler_steps_match_schedule():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = CosineScheduler(
        optimizer,
        base_value=0.1,
        final_value=0.01,
        total_iters=10,
        warmup_iters=2,
    )

    assert len(scheduler.schedule) == 10
    for _ in range(3):
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(scheduler[scheduler.iter])

    assert scheduler[10] == pytest.approx(0.01)
