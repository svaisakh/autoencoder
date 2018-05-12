import torch
import imageio

from torch.nn import functional as F
from contextlib import contextmanager
from IPython.display import Image

def merge(tensors, rows, cols):
    assert rows * cols == len(tensors)
    
    new_tensor = torch.cat(tensors[:cols], 2)
    for i in range(1, rows):
        new_row = torch.cat(tensors[i * rows: i * rows + cols], 2)
        new_tensor = torch.cat([new_tensor, new_row] , 1)
        
    return new_tensor

@contextmanager
def evaluate(*modules):
    states = []
    for module in modules:
        states.append(module.training)
        module.eval()
    with torch.no_grad(): yield
    for i, module in enumerate(modules): module.training = states[i]

def animate_interpolation(model, latent_size, data_valid):
    direction = torch.randn(latent_size)
    
    # Get decoded images with the new latent vectors
    xs = []
    for _ in range(64):
        # Get latent vector of a random validation image
        x = next(iter(data_valid))[0][:1]
        with evaluate(model): z = model.encoder(x)

        # Create copies along the random direction
        zs = torch.linspace(-32, 32, 64).unsqueeze(-1) * direction.unsqueeze(0) * 0.1 + z
    
        with evaluate(model): x_re = F.tanh(model.decoder(zs))
        x_re = x_re.view(x_re.size(0), 28, 28)
        xs.append(x_re)
    
    xs = merge(xs, 8, 8)
    images = list(((xs + 1) * 127.5).byte().numpy())
    fname = str('latent.gif')
    imageio.mimsave(fname, images + images[::-1])
    return Image(filename=fname, width=500)