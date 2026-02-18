import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def normal_velocities(num_vectors, radius=0.1):
  radii = radius * torch.sqrt(torch.rand(num_vectors, dtype=torch.float64))
  angles = 2 * torch.pi * torch.rand(num_vectors)
  x = radii * torch.cos(angles)
  y = radii * torch.sin(angles)
  return torch.stack((x, y), dim=1)


def generate(initial_points,
            inside_criterion,
            immersion,
            process='brownian',
            batch_size=10000,
            n_steps=100,
            v_scale=0.1,
            noise_scale=0.01,
            rtol=1e-10,
            atol=1e-12
            ):

      from .geodesic_solver import Immersed_Manifold
      manifold = Immersed_Manifold(immersion)

      N = initial_points.shape[0]
      V, pos =  torch.zeros((N, n_steps+1, 2), dtype=torch.float64), torch.zeros((N, n_steps+1, 2), dtype=torch.float64)
      V_pt = torch.zeros((N, 2), dtype=torch.float64)
      pos[:,0] = initial_points

      def random_velocities(start_pts, v_scale):
        jacobians = manifold.compute_partial_derivatives(start_pts).transpose(1,2)
        pinvs = torch.linalg.pinv(jacobians)
        Q,_ = torch.linalg.qr(jacobians)
        N = start_pts.shape[0]
        small_vectors = normal_velocities(N, radius=v_scale).unsqueeze(-1)
        random_tangents_3d = torch.bmm(Q, small_vectors)
        random_tangents = torch.bmm(pinvs, random_tangents_3d).squeeze(-1)
        return random_tangents

      V_pt = random_velocities(initial_points, v_scale=v_scale)

      # normalize process name for robustness
      proc = process.lower() if isinstance(process, str) else process

      for i in tqdm(range(n_steps)):
        start_pts = pos[:, i]
        finalized = torch.zeros(N, dtype=torch.bool)
        counter = 0
        jacobians = manifold.compute_partial_derivatives(start_pts).transpose(1,2)
        pinvs = torch.linalg.pinv(jacobians)

        initial_pick = True
        while True:
          unfinalized = ~finalized
          if initial_pick:
            if proc == 'ar1':
              noise = random_velocities(start_pts[unfinalized], v_scale=noise_scale)
              V_prospective = V_pt[unfinalized] + noise
            elif proc == 'brownian':
              V_prospective = random_velocities(start_pts[unfinalized], v_scale=v_scale)
            else:
              raise ValueError(f"Unknown process '{process}'")
            initial_pick = False
          else:
            V_prospective = random_velocities(start_pts[unfinalized], v_scale=v_scale)

          pos_prospective = pos[unfinalized, i] + V_prospective
          inside = inside_criterion(pos_prospective)
          N_inside = inside.sum().item()

          if N_inside > 0:
            finalized_indices = torch.where(unfinalized)[0]
            valid_indices = finalized_indices[inside]
            V[valid_indices,i+1] = V_prospective[inside]
            exp_and_pt = manifold.exp(pos[valid_indices, i], V_prospective[inside], rtol=rtol, atol=atol)
            pos_2d, V_pt_2d = exp_and_pt[:,:2], exp_and_pt[:,2:]

            pos[valid_indices,i+1] = pos_2d.cpu()
            V_pt[valid_indices] = V_pt_2d.cpu()
            (finalized[unfinalized]) = inside

            counter = finalized.sum().item()
          if counter >= N:
              break

      return V[:,1:], pos[:,1:]
