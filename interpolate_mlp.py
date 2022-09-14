import jax
import wandb
import pickle
import haiku as hk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import lerp_params, get_mnist_dataloader
from mlp import BATCH_SIZE, mlp_fn, MNISTBatch

MODEL_A = 0
MODEL_B = 0

BATCH_SIZE = 1024


def load_params(filepath: str):
    """Load model parameters from a file"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    return jax.device_put(params)


def main():
    wandb.init(project='rebasin')

    # Load params for two models with different initialisations
    params_a = load_params(wandb.use_artifact('beetz/rebasin/mnist_mlp_weights:v6').get_path('checkpoint_24_0.pkl').download())
    params_b = load_params(wandb.use_artifact('beetz/rebasin/mnist_mlp_weights:v7').get_path('checkpoint_24_1.pkl').download())

    network = hk.without_apply_rng(hk.transform(mlp_fn))
    test_dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE, train=False)

    @jax.jit
    def evaluate(params: hk.Params, batch: MNISTBatch) -> jnp.ndarray:
        """Evaluation metric (classification accuracy)"""
        logits = network.apply(params, batch.images)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == batch.labels)

    alpha_test_accs = []
    alphas = jnp.linspace(0, 1, 30)

    # Evaluate model while interpolating between params_a and params_b
    for alpha in alphas:
        params = lerp_params(alpha, params_a, params_b)

        test_accs = []
        for images, labels in test_dataloader:
            mnist_batch = MNISTBatch(np.array(images), np.array(labels))
            test_acc = evaluate(params, mnist_batch)
            test_accs.append(test_acc)

        alpha_test_accs.append(np.mean(test_accs))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alphas, alpha_test_accs, label='test')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('$\\alpha$')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Model $A$', 'Model $B$'])
    ax.legend()
    ax.set_title("MNIST MLP Accuracy Interpolation")

    plt.savefig('graphs/interpolate_mlp.png', dpi=300)


if __name__ == '__main__':
    main()
