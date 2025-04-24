
import torch, itertools
from auto_LiRPA.perturbations import PerturbationTopo

def brute_force(A, x, k, upper=True):
    """Enumerate *all* <=k-flip perturbations (small dims only!)."""
    best = -1e9 if upper else 1e9
    dim = x.numel()
    for r in range(k+1):
        for idx in itertools.combinations(range(dim), r):
            x_new = x.clone()
            x_new[list(idx)] = 1 - x_new[list(idx)]
            val = (A @ x_new).item()
            best = max(best, val) if upper else min(best, val)
    return best

def test_topological():
    torch.manual_seed(0)
    A = torch.randn(1, 1, 8)                # (batch,spec,dim)
    x = torch.randint(0, 2, (1, 8)).float() # binary input
    ptb = PerturbationTopo(allowed_swaps=2)
    # mimic LiRPA call
    ub = ptb.concretize(x, A, sign=+1)[0,0].item()
    lb = ptb.concretize(x, A, sign=-1)[0,0].item()
    assert abs(ub - brute_force(A.squeeze(0).squeeze(0), x.squeeze(0), 2, True)) < 1e-4
    assert abs(lb - brute_force(A.squeeze(0).squeeze(0), x.squeeze(0), 2, False)) < 1e-4
