import torch
import torch.nn.functional as F
from lr_scheduler import *

def build_lr_schedulers(optimizers):
    schedulers = []
    for optimizer in optimizers:
        scheduler = WarmupMultiStepLR(optimizer, [50,100,200],
                                      gamma=0.1,
                                      warmup_factor=0.1,
                                      warmup_iters=10,
                                      warmup_method='linear')
        schedulers.append(scheduler)
    return schedulers


def reconstruction_l1_loss(x, logits):
    L1 = F.l1_loss(logits, x)
    L1 += F.mse_loss(logits, x)
    return L1


def compute_inv_mult_quad(x1,x2,eps: float = 1e-7):
    #z_dim = x2.size(-1)
    C = 1
    kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))
    result = kernel.sum() - kernel.diag().sum()

    return result


def compute_kernel(x1,x2):
    D = x1.size(1)
    N = x1.size(0)
    x1 = x1.unsqueeze(-2) # Make it into a column tensor
    x2 = x2.unsqueeze(-3) # Make it into a row tensor
    x1 = x1.expand(N, N, D)
    x2 = x2.expand(N, N, D)
    result = compute_inv_mult_quad(x1, x2)

    return result

def compute_mmd(z):

    prior_z = torch.randn_like(z)
    prior_z__kernel = compute_kernel(prior_z, prior_z)
    z__kernel = compute_kernel(z, z)
    priorz_z__kernel = compute_kernel(prior_z, z)

    mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
    return mmd

