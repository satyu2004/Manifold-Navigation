import torch
model_names = ['lstm'] # 'rnn', 'lstm', 'gru', '2lrnn', 'assisted_gru'
dims = [32] # this is 'd'
surface = 'torus' # 'plane', 'sphere', 'torus'
dataset = 'ar1' # 'brownian', 'ar1'
setups = ['k_50'] # 'basic', 'k_20', 'k_30', 'k_50', 'handpicked/pwsof1pt75'

# Import data
from src.helper import load_data
X0, V, pos = load_data(surface, dataset)


train_split, val_split = 0.8, 0.1
N_trajectories = X0.shape[0]
train_size = int(train_split * N_trajectories)
val_size = int(val_split * N_trajectories)

X_train, V_train, pos_train = X0[:train_size], V[:train_size], pos[:train_size]
X_val, V_val, pos_val = X0[train_size:train_size+val_size], V[train_size:train_size+val_size], pos[train_size:train_size+val_size]
X_test, V_test, pos_test = X0[train_size+val_size:], V[train_size+val_size:], pos[train_size+val_size:]

from src.helper import import_immersion
immersion = import_immersion(surface)

from src.helper import import_model_architecture
model_architecture = import_model_architecture(model_names[0])
model = model_architecture(hidden_size=dims[0])
model.load_state_dict(torch.load(f'results/model_weights/{surface}/{dataset}/{setups[0]}/{model_names[0]}/d_{dims[0]}/run_0.pth'))
model.eval()

pred = model(X_train, V_train)

from src.helper import compute_error
error = compute_error(pred, pos_train, immersion)
print(f"Train Error: {error}")