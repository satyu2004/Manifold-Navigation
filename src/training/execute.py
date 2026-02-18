# Contains the automation of saving model weights and training all models

import src.model_definitions.models as models

import torch
torch.set_default_dtype(torch.float64)
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import numpy as np

import time
from tqdm import tqdm


import matplotlib.pyplot as plt

import os
import pickle

def execute(model_name, surface, hidden_dims, N_trajectories,
            setup='k_50', dataset='brownian', num_layers=2, 
            seq_length=10, agg_indices=[], lr=0.01, gamma=0.995, 
            decay_wt=1.0, num_epochs=1000, batch_size=1024, n_runs=10,
            train_split=0.8, val_split=0.1):
    def pickle_list(my_list, filename):
        """
        Pickles a list and saves it to a file.

        Args:
            my_list (list): The list to pickle.
            filename (str): The filename to save the pickled list to.
        """
        try:
            with open(filename, 'wb') as f:  # 'wb' for write binary
                pickle.dump(my_list, f)
            print(f"List pickled and saved to {filename}")
        except IOError as e:
            print(f"Error pickling list: {e}")
    
    def log_message(message):
        """
        Logs a message to the log file and prints it to the console.

        Args:
            message (str): The message to log.
        """
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + "\n")
        print(message)

    # Initialize log file
    save_path = f'results/model_weights/{surface}/{dataset}/{setup}/{model_name}'
    log_file_path = f'{save_path}/execution_log.txt'

    if os.path.isdir(save_path):
        log_message(f"'{save_path}' exists")
    else:
        os.makedirs(save_path)
        log_message(f"'{save_path}' does not exist. Creating the directory...")
        

    log_message(f'Saving model weights to {save_path}')

    plot_path = f'results/animations/{surface}/{dataset}/{setup}/{model_name}'
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)



    with open(log_file_path, 'w') as log_file:
        log_file.write("Execution Log\n")
        log_file.write("="*50 + "\n")
        log_file.write(f"Model Name: {model_name}\n")
        log_file.write(f"Surface: {surface}\n")
        log_file.write(f"Hidden Dimensions: {hidden_dims}\n")
        log_file.write(f"Number of Trajectories: {N_trajectories}\n")
        log_file.write(f"Setup: {setup}\n")
        log_file.write(f"Dataset: {dataset}\n")
        # log_file.write(f"Number of Layers: {num_layers}\n")
        log_file.write(f"Sequence Length: {seq_length}\n")
        log_file.write(f"Learning Rate: {lr}\n")
        log_file.write(f"Decay Weight: {decay_wt}\n")
        log_file.write(f"Number of Epochs: {num_epochs}\n")
        log_file.write(f"Batch Size: {batch_size}\n")
        log_file.write(f"Number of Runs: {n_runs}\n")
        log_file.write("="*50 + "\n\n")



    # model_architectures = {
    #     'rnn': models.RNN,
    #     'lstm': models.ConditionalLSTM,
    #     'gru': models.ConditionalGRU,
    #     # '2lrnn': models.RNN_multilayer,
    #     # 'assisted_gru': models.assisted_GRU,
    #     # 'GRUWithDecoder': models.GRUWithDecoder,
    #     # 'ResidualGRU': models.ResidualGRU,
    #     # 'DifferenceEncodingGRU': models.DifferenceEncodingGRU,
    #     # 'DualPathGRU': models.DualPathGRU,
    #     # 'ModifiedGateGRU': models.ModifiedGateGRU,
    #     # 'PhysicsInformedGRU': models.PhysicsInformedGRU,
    #     # 'AttentionResidualGRU': models.AttentionResidualGRU
    # }

    # base_architecture = model_architectures.get(model_name)
    # if base_architecture is None:
    #     raise ValueError(f'Invalid model name: {model_name}')
    from src.helper import import_model_architecture
    base_architecture = import_model_architecture(model_name)

    # Import immersion function based on surface type (dynamic)
    from src.helper import import_immersion
    immersion = import_immersion(surface)

    X0 = torch.load(f'src/datasets/{surface}/{dataset}/X0.pt').to(device)[:N_trajectories]
    V = torch.load(f'src/datasets/{surface}/{dataset}/V.pt').to(device)[:N_trajectories]
    pos = torch.load(f'src/datasets/{surface}/{dataset}/pos.pt').to(device)[:N_trajectories]

    # if model_name in ['rnn', 'lstm', 'gru', 'assisted_gru']:
    #     RNN_models = [[base_architecture(hidden_size=d)]*n_runs for d in hidden_dims]
    if model_name == '2lrnn':
        RNN_models = [[base_architecture(hidden_size=d, num_layers=num_layers)]*n_runs for d in hidden_dims]
    else:
        RNN_models = [[base_architecture(hidden_size=d)]*n_runs for d in hidden_dims]


    N = X0.shape[0]
    n = V.shape[1]
    # Data Preparation and Train/Val/Test Splitting
    # train_split = 0.8
    # val_split = 0.1
    # test_split = 0.1
    train_end = int(train_split * N)
    val_end = int((train_split + val_split) * N)

    X0_train = X0[:train_end]
    X0_val = X0[train_end:val_end]
    X0_test = X0[val_end:]
    V_train = V[:train_end]
    V_val = V[train_end:val_end]
    V_test = V[val_end:]
    pos_train = pos[:train_end]
    pos_val = pos[train_end:val_end]
    pos_test = pos[val_end:]

    pos_test_naive = X0_test.unsqueeze(1) + torch.cumsum(V_test, dim=1)
    errors_naive = (immersion(pos_test) - immersion(pos_test_naive)).norm(dim=-1).mean(dim=0)

    strobe = 1
    errors_over_time = torch.zeros(n_runs, num_epochs//strobe, n)

    message = f"Agg_indices = {agg_indices}"
    log_message(message)



    
    def train(net, X0, V, pos, seq_length, indices_to_aggregate=[], lr=lr, gamma=gamma, batch_size = 1024, num_epochs = num_epochs):


        k = seq_length
        # Define optimizer and scheduler
        optimizer = optim.Adam(net.parameters(), lr = lr)  # Example optimizer
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # Load your dataset
        train_loader = DataLoader(TensorDataset(X0, V, pos), batch_size=batch_size, shuffle=True)

        # Training loop
        num_epochs = num_epochs
        run_time = time.time()

        # print(f'indices_to_aggregate = {indices_to_aggregate}')
        # if len(indices_to_aggregate)>0:
        if len(agg_indices)>0:
            L = agg_indices
        else:
            L = range(1, k+1)
        # log_message(f"Training by aggregating on indices {L}")
        if 0 in L:
            L.remove(0)  # Remove 0 if it shows up in L


        

        errors_tracker = torch.zeros(num_epochs//strobe, 100)

        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_model_state = None

        early_stopped = False
        for epoch in tqdm(range(num_epochs)):
                running_loss = 0.0

                for minibatch in train_loader:
                    # Forward pass
                    X = minibatch[0].to(device)
                    V = minibatch[1].to(device)
                    Y = minibatch[2].to(device)
                    n_mb = X.shape[0]
                    loss = 0

                    Yhat = net(X0=X, V=V[:,:max(L)])

                    if 0 in indices_to_aggregate:
                        recon_x0 = net.decoder(net.encoder(X0))  # Reconstruct X0 from the encoder
                        loss += nn.MSELoss()(recon_x0, X0)  # Add
                        # L.remove(0)  # Remove 0 from L if it's in indices_to_aggregate
                    
                    # for i in L:
                    #     criterion = nn.MSELoss()
                    #     loss += (decay_wt**i) * criterion(immersion(Y[:,i-1]), immersion(Yhat[:,i-1]))
                    L = np.array(L)
                    criterion = nn.MSELoss()
                    loss = criterion(immersion(Y[:,L-1]), immersion(Yhat[:,L-1]))


                    # Backward pass and optimization
                    loss /= len(L)
                    loss *= n_mb/N
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                # Validation loss (aggregated MSE)
                with torch.no_grad():
                    val_pred = net(X0=X0_val.to(device), V=V_val.to(device))
                    val_loss = nn.MSELoss()(immersion(pos_val.to(device)), immersion(val_pred)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = net.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    log_message(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                    if best_model_state is not None:
                        net.load_state_dict(best_model_state)
                    early_stopped = True
                    break

                if (epoch+1)%strobe==0:
                    with torch.no_grad():
                        pos_pred = net(X0=X, V=V)
                        errors_tracker[epoch//strobe] = (immersion(Y) - immersion(pos_pred)).norm(dim=-1).mean(dim=0)
                scheduler.step()


        if early_stopped:
            # Fill the remainder of errors_tracker with the last computed value
            last_idx = (epoch)//strobe
            if last_idx > 0:
                last_val = errors_tracker[last_idx-1]
                errors_tracker[last_idx:] = last_val
        else:
            log_message(f"Training ran for all {num_epochs} epochs. Final val loss: {val_loss:.6f}")
        runtime = time.time()-run_time  
        log_message(f'Total runtime: {runtime:.2f} seconds. Iterations/second: {epoch/runtime:.2f}')
        # print(f"Training time = {runtime}")
        return runtime, errors_tracker



    import matplotlib.animation as animation

    # Create a directory to save the gif if it doesn't exist
    gif_save_path = f'{plot_path}/error_evolution.gif'

    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=f'Naive Error', color='black')[0]]
    lines += [ax.plot([], [], label=f'Run {i+1}')[0] for i in range(n_runs)]
    lines += [ax.plot([], [], label=f'Median', color='black', linestyle='--')[0]]
    ax.set_xlim(0, errors_over_time.shape[2])  # Timestep
    ax.set_ylim(0, 2*errors_naive.max().item())  # Error range
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Error')
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        lines[0].set_data(range(n), errors_naive.detach().cpu().numpy())
        for i, line in enumerate(lines[1:-1]):
            line.set_data(range(n), errors_over_time[i, frame].detach().cpu().numpy())
            ax.set_title(f'Error Evolution at Epoch {frame*strobe}')
        median_error = torch.median(errors_over_time[:, frame], dim=0).values
        lines[-1].set_data(range(n), median_error.detach().cpu().numpy())
        return lines



    # Training all models and computing predictions
    runtimes = []
    for dim, model_list in tqdm(zip(hidden_dims, RNN_models)):
        log_message(f'Training models with hidden dimension {dim}')
        times = []
        save_dir = f'{save_path}/d_{dim}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            log_message(f'Created directory: {save_dir}')
        for run, model in tqdm(enumerate(model_list)):
            log_message(f'Run {run+1}/{n_runs}  (hidden_dim={dim})')
            t, errors_tracker = train(model.to(device), X0=X0_train, V=V_train, pos=pos_train, seq_length=seq_length, indices_to_aggregate=[], batch_size = batch_size, num_epochs = num_epochs)
            times.append(t)

            errors_over_time[run] = errors_tracker



            # Save Model Weights
            torch.save(model.state_dict(), f'{save_dir}/run_{run}.pth')

            pos_pred = torch.zeros_like(pos_test)
            # for i in range(n):
            #     pos_pred[:,i] = model(x_0=X0_test, V=V_test[:,:i+1]).squeeze()
            # torch.save(pos_pred, f'results\{base_name}\hidden_dim_{dim}_{run}.pt')
        # pickle_list(times, f'{path}/runtimes/k_{seq_length}/runtimes_{model_name}_dim_{dim}.pkl')
            # Create a directory to save the gif if it doesn't exist
        gif_save_path = f'{plot_path}/error_evolution_{dim}.gif'
        import matplotlib.animation as animation
        ani = animation.FuncAnimation(fig, update, frames=errors_over_time.shape[1], init_func=init, blit=True)
        ani.save(gif_save_path, writer='imagemagick', fps=20)

        log_message(f"Error evolution gif saved to {gif_save_path}")