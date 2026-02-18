import torch
import numpy as np
torch.set_default_dtype(torch.float64)


def immersion(X, radius=np.sqrt(np.pi)):
      " Maps points in the plane to points on the sphere via the inverse of stereographic projection"

      X = X / radius
      x = X[..., 0]
      y = X[..., 1]
      R = x**2 + y**2 + 1
      return radius * torch.stack([2*x/R, 2*y/R, 1-2/R], dim=-1)

# def chart(X):
#   x, y, z = X[:,0], X[:,1], X[:,2]
#   x_coords = x/(1-z)
#   y_coords = y/(1-z)
#   return torch.stack((x_coords, y_coords), dim=1)