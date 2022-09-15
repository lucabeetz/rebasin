import jax
import pickle
import haiku as hk
import torchvision.transforms as transforms
from jax.tree_util import tree_map
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_mnist_dataloader(train: bool = True, batch_size: int = 512) -> DataLoader:
    """Get a dataloader for MNIST"""
    mnist_data = MNIST('./data', download=True, train=train, transform=transforms.ToTensor())
    data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    return data_loader


def lerp_params(alpha, params_a, params_b) -> hk.Params:
    """Linearly interpolate between two sets of parameters"""
    return tree_map(lambda a, b: (1 - alpha) * a + alpha * b, params_a, params_b)


def load_params(filepath: str):
    """Load model parameters from a file"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    return jax.device_put(params)
