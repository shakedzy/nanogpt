from torch import nn


def initialize_weights(module: nn.Module) -> None:
    """
    Initialize weights of the model:
    - Embedding layers: Normal initialization with std=0.02
    - Linear layers in attention: Xavier initialization
    - Linear layers in feedforward with ReLU: Kaiming initialization
    - LayerNorm: Weights set to 1, biases set to 0
    """
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.Linear):
        if hasattr(module, 'activation') and module.activation == 'relu':  
            # Use Kaiming for feedforward layers with ReLU
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        else:
            # Use Xavier for attention layers
            nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
