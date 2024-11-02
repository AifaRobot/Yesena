import numpy as np
import torch
from typing import Callable

def conjugate_gradient(A: Callable, b: torch.Tensor, steps: int, tol: float = 1e-6) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b - A(x)
    d = r.clone()
    tol_new = r.t() @ r
    for _ in range(steps):
        if tol_new < tol:
            break
        q = A(d)
        alpha = tol_new / (d.t() @ q)
        x += alpha * d
        r -= alpha * q
        tol_old = tol_new.clone()
        tol_new = r.t() @ r
        beta = tol_new / tol_old
        d = r + beta * d
    return x

def categorical_kl(p_nk: torch.Tensor, q_nk: torch.Tensor):
    ratio_nk = p_nk / (q_nk + 1e-6)
    ratio_nk[p_nk == 0] = 1
    ratio_nk[(q_nk == 0) & (p_nk != 0)] = np.inf
    return (p_nk * torch.log(ratio_nk)).sum(dim=1)

def get_flat_params_from(model: torch.nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params_to(params: torch.nn.Module.parameters, model: torch.nn.Module):
    pointer = 0
    for p in model.parameters():
        p.data.copy_(params[pointer:pointer + p.data.numel()].view_as(p.data))
        pointer += p.data.numel()

def get_grad_no_flatened(grad_flattened, local_actor_critic):
        n = 0
        grad_no_flatened = []

        for param in local_actor_critic.parameters():
            numel = param.numel()
            g = grad_flattened[n:n + numel].view(param.shape)
            grad_no_flatened.append(g)
            n += numel

        return grad_no_flatened
