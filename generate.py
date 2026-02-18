# Generate trajectories on a manifold of your definition
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

####################################################################################
########################### NAME YOUR SURFACE AND PROCESS HERE##########################
####################################################################################
surface = 'torus' # Name your surface here
process = 'ar1' # Choose your process here. Options: 'brownian', 'ar1'

####################################################################################
########################### DEFINE THE INSIDE CRITERION ##########################
########################### FOR YOUR COORDINATE CHART HERE ##########################
####################################################################################
def inside_criterion(pts):
    """
    Checks if a point x is inside the coordinate chart U.

    Parameters
    ----------
    x : Tensor
        Tensor of shape (..., dim)

    Returns
    -------
    Tensor
        Boolean tensor of shape (...,)
    
    Example Usage
    -----
    Modify this function according to the chart you are using.

    For rectangle with sides [0, a] x [0, b], you can do:
    x_ok = (X > 0) & (X < a)
    y_ok = (Y > 0) & (Y < b)
    return x_ok & y_ok

    For circle of radius r, you can do:
    r = torch.norm(pts, dim=-1)
    return r < radius
    """

    X, Y = pts[..., 0], pts[..., 1]
    
    # Example for rectangular chart:
    a, b = 2*torch.pi, 2*torch.pi
    x_ok = (X > 0) & (X < a)
    y_ok = (Y > 0) & (Y < b)
    return x_ok & y_ok

####################################################################################
########################### GIVE A BOUNDING BOX FOR YOUR ##########################
########################### FOR YOUR COORDINATE CHART HERE ##########################
####################################################################################
# Specify lower left and upper right corners for the box
bounding_box = torch.tensor([[0.0, 0.0], [2*torch.pi, 2*torch.pi]])  # Define the bounding box of the chart U


####################################################################################
########################### DEFINE THE IMMERSION FUNCTION HERE ##########################
####################################################################################
def immersion(pts):
    """
    Immersion map from coordinate chart U to the surface.

    Parameters
    ----------
    pts : Tensor
        Tensor of shape (..., dim)

    Returns
    -------
    Tensor
        Tensor of shape (..., surface_dim)
    
    Example Usage
    -----
    Modify this function according to the immersion you are using.

    For rectangle to torus immersion:
    X = (c + a * cos(y)) * cos(x)
    Y = (c + a * cos(y)) * sin(x)
    Z = a * sin(y)
    return torch.stack((X, Y, Z), dim=-1)
    """
    x, y = pts[..., 0], pts[..., 1]
    
    # Example for torus immersion:
    a, c = 1.0, 4.0  # Torus parameters
    X = (c + a * torch.cos(y)) * torch.cos(x)
    Y = (c + a * torch.cos(y)) * torch.sin(x)
    Z = a * torch.sin(y)
    return torch.stack([X, Y, Z], dim=-1).to(device)



####################################################################################
########################### SPECIFY DATASET PARAMETERS HERE ##########################
####################################################################################

# Dataset size parameters
batch_size = 10000 # How many trajectories do you want? (Batch Size)
n_steps = 100 # How many time steps do you want? (Sequence Length)

# Scale parameters for velocity sampling
v_scale = 0.1 # Scale parameter for velocities
noise_scale = 0.01 # Scale parameter for noise in AR(1) process

# Precision parameters (for ODE solver)
rtol = 1e-7 # Relative tolerance
atol = rtol/100 # Absolute tolerance

####################################################################################
########################### SPECIFY INITIAL POINTS  ##########################
########################### FOR YOUR TRAJECTORIES HERE ##########################
####################################################################################

# Shape must be (batch_size, 2)
X0 = torch.rand((batch_size,2), dtype=torch.float64)*2*torch.pi


####################################################################################
########################### SIT BACK AND ENJOY.  ##################################
##################### YOUR TRAJECTORIES ARE BEING GENERATED ##########################
####################################################################################

# Check if the directory exists, if not, create it
path = f'src/datasets/{surface}/{process}/'
os.makedirs(path, exist_ok=True)
import os
if os.path.exists(path):
    print(f'{path} exists. Saving...')
else:
    print(f'Created directory {path}.')

# import sys, pathlib
# sys.path.insert(0, str(pathlib.Path().resolve() / "src"))

from src.random_generators import generate
V, pos = generate(process=process,
                                initial_points=X0,
                                inside_criterion=inside_criterion,
                                immersion=immersion,
                                batch_size=batch_size,
                                n_steps=n_steps,
                                v_scale=v_scale,
                                noise_scale=noise_scale,
                                rtol=rtol,
                                atol=atol)
  


torch.save(X0, f'{path}/X0.pt')
torch.save(V, f'{path}/V.pt')
torch.save(pos, f'{path}/pos.pt')

print("Done.")
print(f"Trajectories saved to {path}")
print(f"X0.shape:{X0.shape}")
print(f"V.shape:{V.shape}")
print(f"pos.shape:{pos.shape}")