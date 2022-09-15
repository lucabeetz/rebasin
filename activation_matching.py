from matplotlib.pyplot import axis
import wandb
import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mlp import MNISTBatch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from utils import lerp_params, load_params, get_mnist_dataloader

BATCH_SIZE = 60_000
NUM_CLASSES = 10


def mlp_activations_fn(x: jnp.ndarray):
    """MLP with three hidden layers and intermediate activations"""
    activations = {}

    x = hk.Flatten()(x)

    activations['linear'] = hk.Linear(512)(x)
    x = jax.nn.relu(activations['linear'])

    activations['linear_1'] = hk.Linear(512)(x)
    x = jax.nn.relu(activations['linear_1'])

    activations['linear_2'] = hk.Linear(512)(x)
    x = jax.nn.relu(activations['linear_2'])

    activations['linear_3'] = hk.Linear(NUM_CLASSES)(x)

    return activations['linear_3'], activations


def main():
    wandb.init(project='rebasin')

    # Load params for two models with different initialisations
    params_a = load_params(wandb.use_artifact('beetz/rebasin/mnist_mlp_weights:v6').get_path('checkpoint_24_0.pkl').download())
    params_b = load_params(wandb.use_artifact('beetz/rebasin/mnist_mlp_weights:v7').get_path('checkpoint_24_1.pkl').download())

    network = hk.without_apply_rng(hk.transform(mlp_activations_fn))
    train_dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE)

    images, labels = next(iter(train_dataloader))
    mnist_batch = MNISTBatch(np.array(images), np.array(labels))

    # Compute activations for both models, shapes: (BATCH_SIZE, FEATURE_DIM)
    _, activations_a = network.apply(params_a, mnist_batch.images)
    _, activations_b = network.apply(params_b, mnist_batch.images)

    permutations = []

    # Run activation matching for each layer
    for layer in activations_a:
        # Calculate mean activation
        mean_a = activations_a[layer].mean(axis=0)
        mean_b = activations_b[layer].mean(axis=0)

        res_a = activations_a[layer] - mean_a
        res_b = activations_b[layer] - mean_b

        cov = res_a.T @ res_b / (BATCH_SIZE - 1)

        E_ab = cov + jnp.outer(mean_a, mean_b)

        ri, ci = linear_sum_assignment(E_ab, maximize=True)
        permutations.append(ci)


    permuted_params = {}
    for index, layer in enumerate(params_b):
        p = params_b[layer].copy()

        permuted_b = p['b'].take(permutations[index])
        permuted_w = p['w'].take(permutations[index], axis=1)

        if index > 0:
            permuted_w = permuted_w.take(permutations[index - 1], axis=0)

        permuted_params[layer] = {'b': permuted_b, 'w': permuted_w}

    @jax.jit
    def evaluate(params: hk.Params, batch: MNISTBatch) -> jnp.ndarray:
        """Evaluation metric (classification accuracy)"""
        logits, _ = network.apply(params, batch.images)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == batch.labels)

    test_dataloader = get_mnist_dataloader(batch_size=1_000, train=False)

    alpha_naive_accs = []
    alpha_perm_accs = []
    alphas = jnp.linspace(0, 1, 50)

    # Evaluate model on naive and matched permutations
    for alpha in tqdm(alphas):
        naive_params = lerp_params(alpha, params_a, params_b)
        perm_params = lerp_params(alpha, params_a, permuted_params)

        naive_accs = []
        perm_accs = []
        for images, labels in test_dataloader:
            mnist_batch = MNISTBatch(np.array(images), np.array(labels))

            naive_accs.append(evaluate(naive_params, mnist_batch))
            perm_accs.append(evaluate(perm_params, mnist_batch))

        alpha_naive_accs.append(np.mean(naive_accs))
        alpha_perm_accs.append(np.mean(perm_accs))

    # Plot accuracies
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alphas, alpha_naive_accs, label='naive')
    ax.plot(alphas, alpha_perm_accs, label='activation matching')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('$\\alpha$')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Model $A$', 'Model $B$'])
    ax.legend()
    ax.set_title("MNIST MLP Test Accuracy")

    plt.savefig('graphs/interpolate_mlp_activations.png', dpi=300)


if __name__ == '__main__':
    main()
