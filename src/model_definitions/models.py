import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import pdb


def baseline(X0, V):
    """
    Treats the space like a flat plane
    """
    pos_pred = X0.unsqueeze(dim=1) + V.cumsum(dim=1)
    return pos_pred

class RNN(nn.Module):
    def __init__(self, hidden_size, input_size=2, output_size=2, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.encoder = nn.Linear(2*output_size, hidden_size)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity='relu', bias=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, X0, V):
        # Initialize hidden state
        h_0 = self.encoder(X0).unsqueeze(0)

        # Forward pass through RNN
        out = self.rnn(V, h_0)
        # out = self.decoder(out[1])
        out = self.decoder(out[0])
        return out

# class RNN_multilayer(nn.Module):
#     def __init__(self, hidden_size, input_size=2, output_size=2, num_layers=2):
#         super(RNN_multilayer, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # self.encoder = nn.Linear(2*output_size, hidden_size)
#         self.encoder = [nn.Linear(input_size, hidden_size)]*num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity='relu', bias=False, batch_first=True)
#         self.decoder = nn.Linear(hidden_size, output_size)

#     def forward(self, x_0, V):
#         # Initialize hidden state
#         h_0 = torch.stack([self.encoder[i](x_0) for i in range(self.num_layers)], dim=0)

#         # Forward pass through RNN
#         out = self.rnn(V, h_0)
#         out = self.decoder(out[1][-1])

#         return out

class RNN_multilayer(nn.Module):
    def __init__(self, hidden_size, input_size=2, output_size=2, num_layers=2):
        super(RNN_multilayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.encoder = nn.Linear(2*output_size, hidden_size)
        # self.encoder = [nn.Linear(input_size, hidden_size)]*num_layers # original version

        self.encoder = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_layers)])
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,nonlinearity='relu', bias=False, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, X0, V):
        # Initialize hidden state
        h_0 = torch.stack([self.encoder[i](X0) for i in range(self.num_layers)], dim=0)

        # Forward pass through RNN
        out = self.rnn(V, h_0)
        # out = self.decoder(out[1][-1])
        out = self.decoder(out[0])

        return out


class ConditionalLSTM(nn.Module):
    def __init__(self, hidden_size, input_size_x=2, input_size_v=2, output_size=2):
        super(ConditionalLSTM, self).__init__()
        self.encoder = nn.Linear(input_size_x, hidden_size)
        self.lstm = nn.LSTM(input_size_v, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, X0, V):
        # Encode x into initial hidden and cell states
        encoded_x = torch.tanh(self.encoder(X0))  # Using tanh activation
        initial_hidden = encoded_x.unsqueeze(0)  # Shape (1, batch_size, hidden_size)
        initial_cell = torch.zeros_like(initial_hidden)  # Initialize cell state to zeros

        # Pass v through the LSTM with the initialized states
        lstm_out, _ = self.lstm(V, (initial_hidden, initial_cell))

        # Take the output from the last time step
        # last_time_step_output = lstm_out[:, -1, :]
        # output = self.fc(last_time_step_output)
        output = self.decoder(lstm_out)
        return output
    



class ConditionalGRU(nn.Module):
    def __init__(self, hidden_size, input_size_x=2, input_size_v=2, output_size=2):
        super(ConditionalGRU, self).__init__()
        self.encoder = nn.Linear(input_size_x, hidden_size)
        # self.encoder_x = nn.Linear(input_size_x, hidden_size)
        self.gru = nn.GRU(input_size_v, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X0, V):
        # Encode x into initial hidden state
        encoded_x = torch.tanh(self.encoder(X0))
        # encoded_x = torch.tanh(self.encoder_x(X0))
        initial_hidden = encoded_x.unsqueeze(0)

        # Pass v through the GRU with the initialized state
        gru_out, _ = self.gru(V, initial_hidden)

        # Take the output from the last time step
        # last_time_step_output = gru_out[:, -1, :]
        # output = self.fc(last_time_step_output)
        output = self.decoder(gru_out)
        # output = self.fc(gru_out)
        return output
    

class assisted_GRU(nn.Module):
    def __init__(self, hidden_size, input_size_x=2, input_size_v=2, output_size=2):
        super(assisted_GRU, self).__init__()
        self.encoder = nn.Linear(input_size_x, hidden_size)
        self.gru = nn.GRU(input_size_v, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, X0, V):
        # Encode x into initial hidden state
        encoded_x = torch.tanh(self.encoder(X0))
        initial_hidden = encoded_x.unsqueeze(0)

        # Pass v through the GRU with the initialized state
        gru_out, _ = self.gru(V, initial_hidden)

        # Take the output from the last time step
        # last_time_step_output = gru_out[:, -1, :]
        # output = self.fc(last_time_step_output)
        output = self.decoder(gru_out)
        baseline_output = baseline(X0, V)
        output = output + baseline_output
        return output
    


################################### Variants of the GRU ##################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GRUCell(nn.Module):
    """
    GRU Cell implementation from scratch
    
    GRU equations:
    r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # Reset gate
    z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # Update gate  
    n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  # New gate
    h_t = (1 - z_t) * n_t + h_{t-1}  # Hidden state update
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input-to-hidden weights (3 gates: reset, update, new)
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)

        # Hidden-to-hidden weights (3 gates: reset, update, new)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        
        if bias:
            # Input biases
            self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
            # Hidden biases
            self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        std = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                param.data.uniform_(-std, std)
    
    def forward(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one time step
        
        Args:
            input_t: Input at time t, shape (batch_size, input_size)
            hidden: Hidden state at time t-1, shape (batch_size, hidden_size)
            
        Returns:
            new_hidden: Updated hidden state, shape (batch_size, hidden_size)
        """
        # print(f"Input shape: {input_t.shape}, Hidden shape: {hidden.shape}")
        # pdb.set_trace()
        # Compute input-to-hidden transformations
        gi = self.weight_ih(input_t)  # (batch_size, 3*hidden_size)
        if self.bias_ih is not None:
            gi = gi + self.bias_ih

        # Compute hidden-to-hidden transformations  
        gh = self.weight_hh(hidden)  # (batch_size, 3*hidden_size)
        if self.bias_hh is not None:
            gh = gh + self.bias_hh
        # print(f"gi shape: {gi.shape}, gh shape: {gh.shape}")
        # pdb.set_trace()
        
        # Split into gates (each is batch_size x hidden_size)
        i_reset, i_update, i_new = gi.chunk(3, 1)
        h_reset, h_update, h_new = gh.chunk(3, 1)
        
        # Reset gate: controls how much of previous hidden state to forget
        reset_gate = torch.sigmoid(i_reset + h_reset)
        
        # Update gate: controls how much to update the hidden state
        update_gate = torch.sigmoid(i_update + h_update)
        
        # New gate: candidate hidden state
        # Reset gate is applied to the hidden-to-hidden connection
        new_gate = torch.tanh(i_new + reset_gate * h_new)
        
        # Final hidden state: new candidate + previous hidden (residual connection)
        new_hidden = (1 - update_gate) * new_gate + hidden
        
        return new_hidden

class GRU(nn.Module):
    """
    Multi-layer GRU implementation from scratch
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 bias: bool = True, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            # First layer takes input_size, others take hidden_size
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(GRUCell(layer_input_size, hidden_size, bias))
    
    def forward(self, input_seq: torch.Tensor, h0: torch.Tensor = None):
        """
        Forward pass through the GRU
        
        Args:
            input_seq: Input sequence 
                - if batch_first=True: (batch_size, seq_len, input_size)
                - if batch_first=False: (seq_len, batch_size, input_size)
            h0: Initial hidden state (num_layers, batch_size, hidden_size)
                If None, initialized to zeros
                
        Returns:
            output: All hidden states
                - if batch_first=True: (batch_size, seq_len, hidden_size)  
                - if batch_first=False: (seq_len, batch_size, hidden_size)
            hidden: Final hidden state (num_layers, batch_size, hidden_size)
        """
        # print(f"Input sequence shape: {input_seq.shape}")
        # pdb.set_trace()
        if self.batch_first:
            batch_size, seq_len, _ = input_seq.shape
            input_seq = input_seq.transpose(0, 1)  # (seq_len, batch_size, input_size)
        else:
            seq_len, batch_size, _ = input_seq.shape

        # print(f"Input sequence shape after transpose: {input_seq.shape}")
        # pdb.set_trace()

        # Initialize hidden state if not provided
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                           dtype=input_seq.dtype, device=input_seq.device)
        
        # Store outputs for each time step
        outputs = []
        
        # Current hidden states for each layer
        hidden_states = [h0[i] for i in range(self.num_layers)]
        
        # Process each time step
        for t in range(seq_len):
            x_t = input_seq[t]  # (batch_size, input_size)
            
            # Pass through each layer
            for layer in range(self.num_layers):
                # print(f"x_t.shape before layer {layer}: {x_t.shape}")
                # pdb.set_trace()
                hidden_states[layer] = self.cells[layer](x_t, hidden_states[layer])
                x_t = hidden_states[layer]  # Output becomes input to next layer
            
            outputs.append(hidden_states[-1])  # Store output of last layer
        
        # Stack outputs: (seq_len, batch_size, hidden_size)
        output = torch.stack(outputs, dim=0)
        
        # Final hidden states: (num_layers, batch_size, hidden_size)
        final_hidden = torch.stack(hidden_states, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
            
        return output, final_hidden

class GRUWithDecoder(nn.Module):
    """
    GRU with a decoder layer to convert hidden states to outputs
    """
    
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2, 
                 num_layers: int = 1, bias: bool = True, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        
        # Encoder: maps initial input to hidden state
        self.encoder = nn.Linear(input_size, hidden_size)

        # GRU layers
        self.gru = GRU(input_size, hidden_size, num_layers, bias, batch_first)
        
        # Decoder: converts hidden states to output
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, X0: torch.Tensor, V: torch.Tensor):
        """
        Forward pass through GRU + Decoder
        
        Args:
            input_seq: Input sequence 
                - if batch_first=True: (batch_size, seq_len, input_size)
                - if batch_first=False: (seq_len, batch_size, input_size)
            h0: Initial hidden state (num_layers, batch_size, hidden_size)
                
        Returns:
            output: Decoded outputs
                - if batch_first=True: (batch_size, seq_len, output_size)  
                - if batch_first=False: (seq_len, batch_size, output_size)
            hidden_seq: All hidden states (same shape as input but with hidden_size)
            final_hidden: Final hidden state (num_layers, batch_size, hidden_size)
        """
        # Encode X0 to initial hidden state
        h0 = self.encoder(X0).unsqueeze(0)  # (1, batch_size, hidden_size)
        input_seq = V
        # Get hidden state sequence from GRU
        hidden_seq, _ = self.gru(input_seq, h0)
        
        # Decode hidden states to outputs
        # hidden_seq shape: (batch_size, seq_len, hidden_size) or (seq_len, batch_size, hidden_size)
        output = self.decoder(hidden_seq)
        
        return output





class ResidualGRU(nn.Module):
    """
    GRU with residual connection implementing: x_t = x_{t-1} + v_t + f(h_t)
    
    Mathematical formulation:
    - Standard GRU: h_t = GRU(h_{t-1}, v_t)  
    - Output: x_t = x_{t-1} + v_t + W_out * h_t + b_out
    
    The GRU learns only the residual correction f(h_t) = W_out * h_t + b_out
    """
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        self.output_size = output_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.residual_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, X0: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_seq: [batch_size, seq_len, input_size] - input sequence
            x0: [batch_size, output_size] - initial state
        Returns:
            x_seq: [batch_size, seq_len, output_size] - predicted sequence
        """

        x0, v_seq = X0, V

        batch_size, seq_len, _ = v_seq.shape
        
        # GRU processing: h_t = GRU(h_{t-1}, v_t)
        h_seq, _ = self.gru(v_seq)  # [batch_size, seq_len, hidden_size]
        
        # Residual predictions: f(h_t) for each timestep
        residuals = self.residual_layer(h_seq)  # [batch_size, seq_len, output_size]
        
        # Implement x_t = x_{t-1} + v_t + f(h_t) iteratively
        x_seq = []
        x_prev = x0
        
        for t in range(seq_len):
            # First-order approximation: x_{t-1} + v_t
            linear_pred = x_prev + v_seq[:, t, :self.output_size] if v_seq.size(-1) >= self.output_size else x_prev
            
            # Add residual correction: x_t = x_{t-1} + v_t + f(h_t)
            x_t = linear_pred + residuals[:, t, :]
            x_seq.append(x_t)
            x_prev = x_t
            
        return torch.stack(x_seq, dim=1)


class DifferenceEncodingGRU(nn.Module):
    """
    GRU with explicit difference encoding as input.
    
    Mathematical formulation:
    - diff_t = x_{t-1} + v_t (first-order approximation)
    - augmented_input_t = [v_t, x_{t-1}, diff_t]
    - h_t = GRU(h_{t-1}, augmented_input_t)
    - x_t = diff_t + W_out * h_t (residual correction)
    """
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        # Input is [v_t, x_{t-1}, diff_t], so size is input_size + 2*output_size
        augmented_input_size = input_size + 2 * output_size
        self.gru = nn.GRU(augmented_input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, X0: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        x0, v_seq = X0, V
        
        batch_size, seq_len, input_size = V.shape

        x_seq = []
        x_prev = x0
        h_prev = None
        
        for t in range(seq_len):
            v_t = v_seq[:, t, :]
            
            # First-order approximation: diff_t = x_{t-1} + v_t
            if v_t.size(-1) >= self.output_size:
                diff_t = x_prev + v_t[:, :self.output_size]
            else:
                # If v_t has fewer dimensions, pad or project
                diff_t = x_prev + F.pad(v_t, (0, self.output_size - v_t.size(-1)))
            
            # Augmented input: [v_t, x_{t-1}, diff_t]
            augmented_input = torch.cat([v_t, x_prev, diff_t], dim=-1).unsqueeze(1)
            
            # GRU processing
            h_t, h_prev = self.gru(augmented_input, h_prev)
            
            # Output: x_t = diff_t + residual_correction
            residual = self.output_layer(h_t.squeeze(1))
            x_t = diff_t + residual
            
            x_seq.append(x_t)
            x_prev = x_t
            
        return torch.stack(x_seq, dim=1)


class DualPathGRU(nn.Module):
    """
    Dual-path processing: separate linear dynamics and residual modeling.
    
    Mathematical formulation:
    Path 1: linear_pred_t = x_{t-1} + v_t
    Path 2: h_t = GRU(h_{t-1}, [v_t, x_{t-1}]), residual_t = f(h_t)
    Output: x_t = linear_pred_t + residual_t
    """
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        # GRU input is [v_t, x_{t-1}]
        gru_input_size = input_size + output_size
        self.gru = nn.GRU(gru_input_size, hidden_size, batch_first=True)
        self.residual_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.output_size = output_size

    def forward(self, X0: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        x0, v_seq = X0, V
        batch_size, seq_len, input_size = v_seq.shape
        
        x_seq = []
        x_prev = x0
        h_prev = None
        
        for t in range(seq_len):
            v_t = v_seq[:, t, :]
            
            # Path 1: Linear dynamics - first-order approximation
            if v_t.size(-1) >= self.output_size:
                linear_pred = x_prev + v_t[:, :self.output_size]
            else:
                linear_pred = x_prev + F.pad(v_t, (0, self.output_size - v_t.size(-1)))
            
            # Path 2: Residual modeling
            gru_input = torch.cat([v_t, x_prev], dim=-1).unsqueeze(1)
            h_t, h_prev = self.gru(gru_input, h_prev)
            residual = self.residual_network(h_t.squeeze(1))
            
            # Combine paths: x_t = linear_pred_t + residual_t
            x_t = linear_pred + residual
            x_seq.append(x_t)
            x_prev = x_t
            
        return torch.stack(x_seq, dim=1)


class ModifiedGateGRU(nn.Module):
    """
    Custom GRU with additional linear gate for the x_{t-1} + v_t structure.
    
    Mathematical formulation:
    Standard GRU gates:
    - r_t = σ(W_r [h_{t-1}, v_t] + b_r)  # reset gate
    - z_t = σ(W_z [h_{t-1}, v_t] + b_z)  # update gate
    
    Additional linear gate:
    - l_t = σ(W_l [h_{t-1}, v_t] + b_l)  # linear contribution gate
    
    Modified update:
    - x_t = (1-z_t) ⊙ (x_{t-1} + l_t ⊙ v_t) + z_t ⊙ h̃_t
    """
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Standard GRU gates
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # reset gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # update gate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # candidate hidden
        
        # Additional linear gate
        self.W_l = nn.Linear(input_size + hidden_size, output_size)  # linear gate
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, X0: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        x0, v_seq = X0, V
        batch_size, seq_len, input_size = v_seq.shape
        
        x_seq = []
        x_prev = x0
        h_prev = torch.zeros(batch_size, self.hidden_size, device=v_seq.device)
        
        for t in range(seq_len):
            v_t = v_seq[:, t, :]
            
            # Concatenate hidden and input for gates
            combined = torch.cat([h_prev, v_t], dim=-1)
            
            # Standard GRU computations
            r_t = torch.sigmoid(self.W_r(combined))  # reset gate
            z_t = torch.sigmoid(self.W_z(combined))  # update gate
            
            # Candidate hidden state
            combined_reset = torch.cat([r_t * h_prev, v_t], dim=-1)
            h_tilde = torch.tanh(self.W_h(combined_reset))
            
            # Update hidden state
            h_t = (1 - z_t) * h_prev + z_t * h_tilde
            
            # Linear contribution gate
            l_t = torch.sigmoid(self.W_l(combined))
            
            # Modified output combining linear dynamics and hidden state
            if v_t.size(-1) >= self.output_size:
                linear_component = x_prev + l_t * v_t[:, :self.output_size]
            else:
                padded_v = F.pad(v_t, (0, self.output_size - v_t.size(-1)))
                linear_component = x_prev + l_t * padded_v
                
            hidden_component = self.output_proj(h_t)
            
            # Combine with learned mixing (using update gate)
            z_out = z_t[:, :self.output_size] if z_t.size(-1) >= self.output_size else z_t.mean(dim=-1, keepdim=True).expand(-1, self.output_size)
            x_t = (1 - z_out) * linear_component + z_out * hidden_component
            
            x_seq.append(x_t)
            x_prev = x_t
            h_prev = h_t
            
        return torch.stack(x_seq, dim=1)


class PhysicsInformedGRU(nn.Module):
    """
    Standard GRU with physics-informed loss capability.
    
    The physics constraint is: x_t ≈ x_{t-1} + v_t + residual(x_{t-1}, v_t)
    This constraint is enforced through a custom loss function.
    """
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Residual function: maps [x_{t-1}, v_t] -> expected residual
        self.residual_predictor = nn.Sequential(
            nn.Linear(output_size + input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x0: torch.Tensor, v_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both predictions and physics constraint violations for loss computation.
        """
        # Standard GRU forward pass
        h_seq, _ = self.gru(v_seq)
        x_pred = self.output_layer(h_seq)
        
        # Compute physics constraint violations for loss
        batch_size, seq_len, _ = v_seq.shape
        physics_violations = []
        x_prev = x0
        
        for t in range(seq_len):
            v_t = v_seq[:, t, :]
            x_t_pred = x_pred[:, t, :]
            
            # Expected residual from physics model
            if v_t.size(-1) >= x0.size(-1):
                residual_input = torch.cat([x_prev, v_t[:, :x0.size(-1)]], dim=-1)
            else:
                padded_v = F.pad(v_t, (0, x0.size(-1) - v_t.size(-1)))
                residual_input = torch.cat([x_prev, padded_v], dim=-1)
                
            expected_residual = self.residual_predictor(residual_input)
            
            # Physics constraint: x_t should equal x_{t-1} + v_t + expected_residual
            if v_t.size(-1) >= x0.size(-1):
                physics_pred = x_prev + v_t[:, :x0.size(-1)] + expected_residual
            else:
                physics_pred = x_prev + F.pad(v_t, (0, x0.size(-1) - v_t.size(-1))) + expected_residual
                
            violation = x_t_pred - physics_pred
            physics_violations.append(violation)
            x_prev = x_t_pred
            
        physics_violations = torch.stack(physics_violations, dim=1)
        return x_pred, physics_violations
    
    def physics_informed_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor, 
                             physics_violations: torch.Tensor, lambda_physics: float = 0.1) -> torch.Tensor:
        """
        Combined loss: L = L_prediction + λ * L_physics
        """
        prediction_loss = F.mse_loss(x_pred, x_true)
        physics_loss = torch.mean(physics_violations ** 2)
        return prediction_loss + lambda_physics * physics_loss


class AttentionResidualGRU(nn.Module):
    """
    GRU with attention mechanism focusing on when/where residual corrections are needed.
    
    Mathematical formulation:
    - h_t = GRU(h_{t-1}, v_t)
    - attention_weights = softmax(W_att * [x_{t-1}, v_t])
    - residual_correction = attention_weights ⊙ f(h_t)
    - x_t = x_{t-1} + v_t + residual_correction
    """
    def __init__(self, hidden_size: int, input_size: int = 2, output_size: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        # Attention mechanism for residual importance
        attention_input_size = output_size + input_size
        self.attention = nn.Sequential(
            nn.Linear(attention_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=-1)
        )
        
        # Residual prediction
        self.residual_layer = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        
    def forward(self, X0: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        x0, v_seq = X0, V
        batch_size, seq_len, input_size = v_seq.shape
        
        # GRU processing
        h_seq, _ = self.gru(v_seq)
        
        x_seq = []
        x_prev = x0
        
        for t in range(seq_len):
            v_t = v_seq[:, t, :]
            h_t = h_seq[:, t, :]
            
            # Attention weights based on current state and input
            if v_t.size(-1) >= self.output_size:
                attention_input = torch.cat([x_prev, v_t[:, :self.output_size]], dim=-1)
            else:
                padded_v = F.pad(v_t, (0, self.output_size - v_t.size(-1)))
                attention_input = torch.cat([x_prev, padded_v], dim=-1)
                
            attention_weights = self.attention(attention_input)
            
            # Residual correction with attention
            residual_raw = self.residual_layer(h_t)
            residual_correction = attention_weights * residual_raw
            
            # Output: x_t = x_{t-1} + v_t + attended_residual
            if v_t.size(-1) >= self.output_size:
                x_t = x_prev + v_t[:, :self.output_size] + residual_correction
            else:
                x_t = x_prev + F.pad(v_t, (0, self.output_size - v_t.size(-1))) + residual_correction
                
            x_seq.append(x_t)
            x_prev = x_t
            
        return torch.stack(x_seq, dim=1)


