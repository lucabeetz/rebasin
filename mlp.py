import jax
import optax
import wandb
import pickle
import haiku as hk
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Tuple
from tqdm import tqdm
from utils import get_mnist_dataloader

LEARNING_RATE = 0.1
BATCH_SIZE = 1024
MAX_EPOCHS = 25
RANDOM_SEED = 1

NUM_CLASSES = 10


class MNISTBatch(NamedTuple):
    images: np.ndarray
    labels: np.ndarray


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def mlp_fn(images: jnp.ndarray):
    """MLP with three hidden layers"""
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(512), jax.nn.relu,
        hk.Linear(512), jax.nn.relu,
        hk.Linear(512), jax.nn.relu,
        hk.Linear(NUM_CLASSES)
    ])

    return mlp(images)


def main():
    # Save run config to wandb
    config = {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'seed': RANDOM_SEED
    }
    wandb.init(project='rebasin', config=config)

    train_dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE)
    test_dataloader = get_mnist_dataloader(batch_size=BATCH_SIZE, train=False)

    # Create the model
    network = hk.without_apply_rng(hk.transform(mlp_fn))

    # Create learning rate schedule and optimiser
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=LEARNING_RATE,
        warmup_steps=10,
        decay_steps=len(train_dataloader.dataset) * MAX_EPOCHS
    )
    optimiser = optax.sgd(lr_schedule, momentum=0.9)

    def loss_fn(params: hk.Params, batch: MNISTBatch) -> jnp.ndarray:
        """Cross-entropy classification loss"""
        batch_size, *_ = batch.images.shape
        logits = network.apply(params, batch.images)
        labels = jax.nn.one_hot(batch.labels, NUM_CLASSES)

        log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))
        return -log_likelihood / batch_size

    @jax.jit
    def evaluate(params: hk.Params, batch: MNISTBatch) -> jnp.ndarray:
        """Evaluation metric (classification accuracy)"""
        logits = network.apply(params, batch.images)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == batch.labels)

    @jax.jit
    def update(state: TrainingState, batch: MNISTBatch) -> Tuple[TrainingState, jnp.ndarray]:
        """Parameter update step on one batch"""
        loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
        updates, opt_state = optimiser.update(grads, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), loss

    # Initialise parameters and optimiser state
    images, labels = next(iter(train_dataloader))
    initial_params = network.init(jax.random.PRNGKey(RANDOM_SEED), np.array(images))
    initial_opt_state = optimiser.init(initial_params)
    state = TrainingState(initial_params, initial_opt_state)

    pbar = tqdm(range(MAX_EPOCHS))
    for epoch in pbar:
        # Training loop
        train_losses = []
        for images, labels in train_dataloader:
            mnist_batch = MNISTBatch(np.array(images), np.array(labels))
            state, loss = update(state, mnist_batch)
            train_losses.append(loss)

        train_loss = np.mean(train_losses)

        # Evaluate model on test set
        test_accs = []
        for images, labels in test_dataloader:
            mnist_batch = MNISTBatch(np.array(images), np.array(labels))
            acc = evaluate(state.params, mnist_batch)
            test_accs.append(acc)

        test_acc = np.mean(test_accs)

        # Update progress bar
        pbar.set_description(f"Epoch: {epoch}, train/loss: {train_loss:.2f}, test/acc: {test_acc:.2f}")

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'test/acc': test_acc,
        })

    # Save params as artifact to wandb
    params = jax.device_get(state.params)
    artifact = wandb.Artifact('mnist_mlp_weights', type='model_weights')
    with artifact.new_file(f'checkpoint_{epoch}_{RANDOM_SEED}.pkl', mode='wb') as f:
        pickle.dump(params, f)
    wandb.log_artifact(artifact)


if __name__ == '__main__':
    main()
