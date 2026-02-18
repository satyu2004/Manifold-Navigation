import torch
torch.set_default_dtype(torch.float32)
# torch.set_default_dtype(torch.float64)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def model_tanh(hidden_dims: list):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, hidden_dims[0]),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dims[1], 1)
    )
    return model

f = model_tanh([32, 32]).to(device)
# f.load_state_dict(torch.load('Recon_Surface\\neural_surface.pth'))
path = 'C:/Users/sathy/Documents/Programming/ARL-related/maninav_new/maninav_new/maninav_new/math_functions/neural_surface.pth'
f.load_state_dict(torch.load(path))

def immersion(point):
    """Immersion. Input is a batch of 2-tupes in the unit square."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = point[..., 0]
    Y = point[..., 1]
    Z = f(point).squeeze(-1)  

    return torch.stack(
        [X, Y, Z],
        axis=-1,
    ).to(device)