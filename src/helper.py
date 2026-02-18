import torch
def load_data(surface, dataset, N_trajectories=None):
    X0 = torch.load(f'src/datasets/{surface}/{dataset}/X0.pt')
    V = torch.load(f'src/datasets/{surface}/{dataset}/V.pt')
    pos = torch.load(f'src/datasets/{surface}/{dataset}/pos.pt')
    if N_trajectories is not None:
        X0 = X0[:N_trajectories]
        V = V[:N_trajectories]
        pos = pos[:N_trajectories]
    return X0, V, pos

def import_immersion(surface):
    import importlib
    try:
        immersion_module = importlib.import_module(f"src.math_functions.{surface}_math")
        return immersion_module.immersion
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Invalid surface: {surface}")
    

def import_model_architecture(model_name):
    import src.model_definitions.models as models
    model_architectures = {
        'rnn': models.RNN,
        'lstm': models.ConditionalLSTM,
        'gru': models.ConditionalGRU,
        # '2lrnn': models.RNN_multilayer,
        # 'assisted_gru': models.assisted_GRU,
        # 'GRUWithDecoder': models.GRUWithDecoder,
        # 'ResidualGRU': models.ResidualGRU,
        # 'DifferenceEncodingGRU': models.DifferenceEncodingGRU,
        # 'DualPathGRU': models.DualPathGRU,
        # 'ModifiedGateGRU': models.ModifiedGateGRU,
        # 'PhysicsInformedGRU': models.PhysicsInformedGRU,
        # 'AttentionResidualGRU': models.AttentionResidualGRU
    }

    base_architecture = model_architectures.get(model_name)
    if base_architecture is None:
        raise ValueError(f'Invalid model name: {model_name}')
    return base_architecture

def baseline(X0, V):
    return X0.unsqueeze(1) + V.cumsum(dim=1)

def compute_error(pred, true, immersion):
    pred_pos = immersion(pred)
    true_pos = immersion(true)
    return torch.norm(pred_pos - true_pos, dim=-1).mean(dim=0)