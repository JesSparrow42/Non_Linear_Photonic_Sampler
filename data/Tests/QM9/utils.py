import torch
from torch import autograd

def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Performs (optionally hard) Gumbel-Softmax sampling.
    """
    g = sample_gumbel(logits.size(), device=logits.device)
    y = logits + g
    y = torch.softmax(y / tau, dim=dim)
    if hard:
        # Straight-through
        index = y.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y).scatter_(dim, index, 1.0)
        y = (y_hard - y).detach() + y
    return y

def compute_gradient_penalty(D, real_samples, fake_samples, device='cpu', lambda_gp=10.0):
    """
    Gradient penalty for WGAN‑GP.

    real_samples and fake_samples are tuples: (adjacency, node_features).
    Both tensors must have requires_grad=False when passed.
    """
    real_adj,  real_feat  = real_samples
    fake_adj,  fake_feat  = fake_samples

    batch_size = real_adj.size(0)

    # Sample uniformly between real and fake (separate shapes for adj and features)
    alpha_adj  = torch.rand(batch_size, 1, 1, 1, device=device)   # (B,1,1,1) → broadcast to adjacency (B,N,N,R)
    alpha_feat = torch.rand(batch_size, 1, 1,    device=device)   # (B,1,1)   → broadcast to features  (B,N,F)

    interp_adj  = real_adj  + alpha_adj  * (fake_adj  - real_adj)
    interp_feat = real_feat + alpha_feat * (fake_feat - real_feat)

    interp_adj.requires_grad_(True)
    interp_feat.requires_grad_(True)

    d_interpol = D(interp_adj, interp_feat)
    grad_outputs = torch.ones_like(d_interpol, device=device)

    grads = autograd.grad(
        outputs=d_interpol,
        inputs=[interp_adj, interp_feat],
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    grad_adj, grad_feat = grads
    grad_flat = torch.cat(
        [grad_adj.view(batch_size, -1),
         grad_feat.view(batch_size, -1)],
        dim=1
    )

    penalty = ((grad_flat.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return penalty

def sample_z(batch_size, z_dim, device='cpu'):
    return torch.randn(batch_size, z_dim, device=device)