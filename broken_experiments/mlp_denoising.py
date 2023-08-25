import jax 
import jax.numpy as jnp
import jax.random as random

import numpy as np

import haiku as hk
import optax

from emlp.groups import O
from emlp.reps import V

import emlp.nn.haiku as ehk

import matplotlib.pyplot as plt

from utils import load_dcd_dataset
from ecv.utils.training import fit

EPOCHS = 5
N_MOLECULES = 22
HIDDEN = 128

class MLP(hk.Module):
    def __init__(self):
        super().__init__()
        self.mlp = hk.Sequential([
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(N_MOLECULES*3),
            jax.nn.tanh
        ])
    def __call__(self, x):
        return self.mlp(x)

@hk.transform
def mlp(inputs):
  model = MLP()
  return model(inputs)

emlp = hk.without_apply_rng(hk.transform(ehk.EMLP(N_MOLECULES*V, N_MOLECULES*V, O(3), ch=HIDDEN, num_layers=1)))

def compute_loss_mlp(params, key, x, y):
    y_hat = mlp.apply(params,key, x)

    loss = jnp.abs((y_hat.reshape(y.shape ) - y)).mean()

    return loss

def compute_loss_emlp(params, key, x, y):
    y_hat = emlp.apply(params, x)

    loss = jnp.abs((y_hat.reshape(y.shape ) - y)).mean()

    return loss

def process_mlp_batch(key, batch):
    _, x, _ = batch
    x = x.squeeze(0).flatten()
    noise = random.normal(key, shape=x.shape)
    x_in = x + noise
    return x_in, x

def plot_training(mlp_data, emlp_data):
    iters = np.linspace(1, len(mlp_data.losses), num=len(mlp_data.losses))
    epochs = np.linspace(1, len(mlp_data.epoch_loss), num=len(mlp_data.epoch_loss))

    # plot iters losses 
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(iters, mlp_data.losses)
    ax.plot(iters, emlp_data.losses) 
    fig.savefig('figs/iter_losses.png') 
    plt.close(fig)

    # plot epoch losses
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(epochs, mlp_data.epoch_loss)
    ax.plot(epochs, emlp_data.epoch_loss) 
    fig.savefig('figs/epoch_losses.png') 
    plt.close(fig)

if __name__ == '__main__':
    train_loader = load_dcd_dataset('./data/adp-vacuum.pdb', './data/traj4.dcd', 1)

    key = random.PRNGKey(330)
 
    print(10*'-'+' training mlp ' +'-'*10)
    
    mlp_optimizer = optax.adam(learning_rate=1e-5)

    batch = next(iter(train_loader))
    x, _ = process_mlp_batch(key, batch)

    initial_params = mlp.init(key, x)

    mlp_data = fit(key, initial_params, mlp_optimizer, compute_loss_mlp, process_mlp_batch, train_loader, EPOCHS)
    
    print(10*'-'+' training emlp ' +'-'*10)

    emlp_optimizer = optax.adam(learning_rate=1e-5)

    batch = next(iter(train_loader))
    x, _ = process_mlp_batch(key, batch)

    initial_params = emlp.init(key, x)

    emlp_data = fit(key, initial_params, emlp_optimizer, compute_loss_emlp, process_mlp_batch, train_loader, EPOCHS)
    
    plot_training(mlp_data, emlp_data) 



