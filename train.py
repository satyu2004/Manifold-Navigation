from src.training.execute import execute


model_names = ['lstm'] # 'rnn', 'lstm', 'gru', '2lrnn', 'assisted_gru'
hidden_dim = [32] # this is 'd'
surface = 'torus' # 'plane', 'sphere', 'torus'
dataset = 'ar1' # 'brownian', 'ar1'
train_loss = ['k_50'] # 'basic', 'k_20', 'k_30', 'k_50', 'handpicked/pwsof1pt75'
# setup = 'exp_5' # 'basic', 'k_20', 'k_30', 'k_50', 'handpicked/pwsof1pt75'
# agg_indices  = list(range(1,51)) # use [] if not hand-picking 

N_trajectories = 10000
lr = 0.01
gamma = 0.995
num_epochs = 500
train_split, val_split = 0.8, 0.1
n_runs = 1 # number of distinct models trained
minibatch_size = 4000

indices_dict = {
                'k_10': list(range(1,11)),
                'k_30': list(range(1,31)),
                'k_50': list(range(1,51)),
                'exp_5': [1, 1, 3, 7, 13, 26, 50],
                'exp_10': [1, 1, 2, 2, 4, 5, 8, 12, 17, 24, 35, 50],
                'exp_15': [1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 11, 14, 18, 24, 30, 39, 50],
                'lin_5': [1, 9, 17, 25, 33, 41, 50],
                'lin_10': [1, 5, 9, 14, 18, 23, 27, 32, 36, 41, 45, 50],
                'lin_15': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 50],
                'mixed_3_3': [ 1,  2,  3, 19, 34, 50],
                'mixed_5_5': [ 1,  2,  3,  4,  5, 14, 23, 32, 41, 50],
                'mixed_7_7': [ 1,  2,  3,  4,  5,  6,  7, 14, 20, 26, 32, 38, 44, 50],
                'inc_gaps':[1, 2, 4, 7, 11, 16, 22, 29, 37, 46],
                'primes':[1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47],
                }






for model_name in model_names:
    for loss in train_loss:
        execute(model_name=model_name,
                hidden_dims=hidden_dim,
                surface=surface,
                dataset=dataset,
                setup=loss,
                # agg_indices=agg_indices,
                agg_indices=indices_dict[loss],
                N_trajectories=N_trajectories,
                lr=lr, gamma=gamma,
                num_epochs=num_epochs,
                train_split=train_split, val_split=val_split,
                batch_size=minibatch_size,
                n_runs=n_runs)