import torch

from phase4.selfloop import set_lr


def _pair(lr=3e-4, warmup=500):
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    return m, opt, sched


def test_set_lr_survives_scheduler_steps_and_state_reload():
    m, opt, sched = _pair()
    for _ in range(600):
        opt.step(); sched.step()
    assert abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12
    set_lr(opt, sched, 1e-5)
    assert abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-12
    opt.step(); sched.step()
    assert abs(opt.param_groups[0]["lr"] - 1e-5) < 1e-12
    # a resume loads the saved (already lowered) state and applies the option again
    m2, opt2, sched2 = _pair()
    opt2.load_state_dict(opt.state_dict()); sched2.load_state_dict(sched.state_dict())
    set_lr(opt2, sched2, 1e-5)
    opt2.step(); sched2.step()
    assert abs(opt2.param_groups[0]["lr"] - 1e-5) < 1e-12


def test_set_lr_keeps_adam_moments():
    m, opt, sched = _pair()
    m(torch.ones(1, 2)).sum().backward(); opt.step(); sched.step()
    before = {k: v["exp_avg"].clone() for k, v in opt.state.items()}
    set_lr(opt, sched, 1e-5)
    assert all(torch.equal(before[k], opt.state[k]["exp_avg"]) for k in before)
