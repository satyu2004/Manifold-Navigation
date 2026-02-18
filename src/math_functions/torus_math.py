import torch
torch.set_default_dtype(torch.float64)


def immersion(point, a=1.0, c=4.0):
      """Immersion."""
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      x = point[..., 0]
      y = point[..., 1]
      X = (c + a * torch.cos(y)) * torch.cos(x)
      Y = (c + a * torch.cos(y)) * torch.sin(x)
      Z = a * torch.sin(y)
      return torch.stack(
      [X, Y, Z],
      axis=-1,
      ).to(device)