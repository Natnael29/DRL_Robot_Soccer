# agents/mlp_networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A generic Multi-Layer Perceptron.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256), activation=nn.ReLU, final_activation=None):
        """
        Initialize the MLP.
        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.
            hidden_dims (tuple): Tuple of integers representing the dimensions of hidden layers.
            activation (nn.Module): Activation function to use for hidden layers.
            final_activation (nn.Module, optional): Activation function for the output layer. Defaults to None (linear).
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.final_activation = final_activation

        layers = []
        dims = [input_dim] + list(hidden_dims)
        
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        
        layers.append(nn.Linear(dims[-1], output_dim))
        
        if final_activation is not None:
            layers.append(final_activation())
            
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example usage (for testing the network structure)
if __name__ == '__main__':
    input_dim = 54 # Example observation dimension per agent
    output_dim_actor = 2 # Example for continuous actions like linear_vel, angular_vel
    output_dim_critic = 1 # Example for Q-value

    print("Testing MLP networks...")

    # Actor network example
    actor_net = MLP(input_dim=input_dim, 
                    output_dim=output_dim_actor, 
                    hidden_dims=(256, 256), 
                    activation=nn.ReLU, 
                    final_activation=nn.Tanh) # Tanh scales output to [-1, 1]

    # Critic network example
    # For MADDPG, the critic typically takes concatenated states AND actions of ALL agents.
    # The input_dim for the critic depends on this.
    # For a simplified network definition here, we'll show a basic MLP structure.
    # The actual input dimension for the critic will be calculated and set in the MADDPG agent class.
    
    # Assume total state dim for N agents = N * input_dim
    # Assume total action dim for N agents = N * output_dim_actor
    # Critic input_dim = (N * input_dim) + (N * output_dim_actor)
    # Let's use dummy input sizes to test the MLP structure itself.
    
    dummy_critic_input_dim = 100 # Example combined input dimension
    critic_net = MLP(input_dim=dummy_critic_input_dim, 
                     output_dim=output_dim_critic, 
                     hidden_dims=(256, 256), 
                     activation=nn.ReLU)

    print("MLP networks defined.")
    
    # Dummy input tensor for testing actor
    dummy_actor_input = torch.randn(1, input_dim) # Batch size 1
    dummy_actor_output = actor_net(dummy_actor_input)
    print(f"Actor output shape (batch size 1): {dummy_actor_output.shape}") 
    
    # Dummy input tensor for testing critic
    dummy_critic_input = torch.randn(1, dummy_critic_input_dim)
    dummy_critic_output = critic_net(dummy_critic_input)
    print(f"Critic output shape (batch size 1): {dummy_critic_output.shape}") 
