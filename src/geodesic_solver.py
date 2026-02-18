import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

class Immersed_Manifold:
    def __init__(self, immersion, chart=None):
        self.immersion = immersion
        self.chart = chart

    def compute_partial_derivatives(self, pts):
        pts.requires_grad_(True)
        immersed_pts = self.immersion(pts)
        jac = torch.stack([torch.autograd.grad(immersed_pts[:, i], pts, torch.ones_like(immersed_pts[:, i]), create_graph=True)[0] for i in range(3)], dim=1).transpose(1,2)
        return jac

    def compute_metric_tensor(self, pts):
        x_uv = self.compute_partial_derivatives(pts)
        G = torch.einsum('nid,njd->nij', x_uv, x_uv)
        return G

    def compute_inverse_metric_tensor(self, pts):
        G = (self.compute_metric_tensor(pts)).detach()
        det_G = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        inv_det_G = 1.0 / det_G
        g11 = G[:, 0, 0]
        g12 = G[:, 0, 1]
        g22 = G[:, 1, 1]
        return torch.stack([torch.stack([g22, -g12], dim=-1) * inv_det_G.unsqueeze(-1),
                            torch.stack([-g12, g11], dim=-1) * inv_det_G.unsqueeze(-1)], dim=-2)

    def compute_christoffel_symbols(self, pts):
        pts.requires_grad_(True)
        G = self.compute_metric_tensor(pts)
        G_inv = self.compute_inverse_metric_tensor(pts)

        dG = torch.zeros(pts.shape[0], 2, 2, 2, device=pts.device, dtype=pts.dtype)
        indices = [(i, j) for i in range(2) for j in range(2)]
        for idx, (i, j) in enumerate(indices):
            g_grad = torch.autograd.grad(G[:, i, j].sum(), pts, create_graph=True)[0]
            dG[:, :, i, j] = g_grad.detach()

        term1 = torch.einsum('nkl,nijl->nkij', G_inv, dG)
        term2 = torch.einsum('nkl,njil->nkij', G_inv, dG)
        term3 = torch.einsum('nkl,nlij->nkij', G_inv, dG)
        Gamma = 0.5 * (term1 + term2 - term3)
        return Gamma

    def geodesic_rhs(self, t, Z):
        positions = Z[:, :2]
        velocities = Z[:, 2:]
        G = self.compute_christoffel_symbols(positions)
        accelerations = -torch.einsum('nkij,ni,nj->nk', G, velocities, velocities)
        rhs = torch.cat([velocities, accelerations], dim=1)
        if torch.isnan(rhs).any():
            raise ValueError("rhs contains NaNs")
        if torch.isinf(rhs).any():
            raise ValueError("rhs contains infinities")
        if torch._is_zerotensor(rhs):
            raise ValueError("rhs is a zero tensor")
        return rhs

    def exp(self, base_pts, velocities, rtol=1e-10, atol=1e-12):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_pts = base_pts.to(device)
        velocities = velocities.to(device)
        initial_state = torch.cat([base_pts, velocities], dim=-1).to(device)
        t_span = torch.tensor(np.arange(0, 1, 0.1), dtype=torch.float64).to(device)
        solution = odeint(self.geodesic_rhs, initial_state, t_span, rtol=rtol, atol=atol)
        return solution[-1]
