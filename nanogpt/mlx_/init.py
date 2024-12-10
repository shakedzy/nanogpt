from mlx import nn
from mlx.nn import init


def initialize_weights(module_name: str, module: nn.Module) -> None:
    # to be used with nn.Module.apply_to_modules
    if module.children():
        return None  # only apply init to leaf modules
    elif 'embedding' in module_name:
        module.apply(init.normal(mean=0.0, std=0.02))
    elif 'ln' in module_name:  # LayerNorm
        module.apply(
            map_fn=init.constant(1.), 
            filter_fn=lambda _, name, __: name != 'bias')
        module.apply(
            map_fn=init.constant(0.), 
            filter_fn=lambda _, name, __: name == 'bias')
    else:  # nn.Linear
        if 'ffn.layers.0' in module_name:
            # These are the linear layers with ReLU activation
            module.apply(
                map_fn=lambda a: init.he_uniform(a.dtype)(a, 'fan_in', 1.),
                filter_fn=lambda _, name, __: name != 'bias')
        else:
            module.apply(
                map_fn=lambda a: init.glorot_uniform(a.dtype)(a, 1.), 
                filter_fn=lambda _, name, __: name != 'bias')
        module.apply(
            map_fn=init.constant(0.), 
            filter_fn=lambda _, name, __: name == 'bias')
        