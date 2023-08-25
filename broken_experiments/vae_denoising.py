import jax 
import jax.numpy as jnp
import jax.random as random

import numpy as np

import haiku as hk
import optax

from emlp.groups import SO
from emlp.reps import V, T

import emlp.nn.haiku as ehk

import matplotlib.pyplot as plt

from utils import load_dcd_dataset
from ecv.utils.training import fit




EPOCHS = 50
N_MOLECULES = 22
HIDDEN = 256
LATENT = 3

class AE(hk.Module):
    def __init__(self):
        super().__init__()
        self.encoder = hk.Sequential([
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(LATENT * 2),
            jax.nn.sigmoid,
        ]) 
        self.decoder = hk.Sequential([
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(N_MOLECULES * 3),
        ])
    def __call__(self, x, key):

        initial_z = self.encoder(x)
        mean = initial_z[:LATENT]
        var = initial_z[LATENT:]
        std = jnp.exp(var * 0.5)
        epsilon = random.normal(key, shape=std.shape)

        z = mean + std * epsilon

        out = self.decoder(z)

        return out, mean, var

class MLP(hk.Module):
    def __init__(self):
        super().__init__()
        self.mlp = hk.Sequential([
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(N_MOLECULES*3),
        ])
    def __call__(self, x):
        return self.mlp(x)

def build_ae():
    @hk.transform
    def ae(inputs, key):
        model = AE()
        return model(inputs, key)
    
    def compute_loss_ae(params, key, x, y, testing=False):

            y_hat, mean, var = ae.apply(params,key, x, key)

            #bce = BinaryCrossentropy()
            #reconstrucion = bce(y, y_hat)
            reconstruction = jnp.abs((y_hat.reshape(y.shape ) - y)).mean()
            kl_divergence = -0.5 * jnp.sum(1 + var -jnp.power(mean, 2) - jnp.exp(var))

            loss = reconstruction + kl_divergence
            if testing: return reconstruction
            return loss
        
    return ae, compute_loss_ae

def build_eae():
    eae_encoder = hk.without_apply_rng(hk.transform(ehk.EMLP(N_MOLECULES * V,2*T(1), SO(3), ch=HIDDEN, num_layers=1)))

    @hk.without_apply_rng
    @hk.transform
    def eae_decoder(inputs):
        model = MLP()
        return model(inputs)

    def init_eae(key, x):
        key, subkey = random.split(key)

        e_initial_params = eae_encoder.init(key, x)
        z = jnp.ones(LATENT) 

        d_initial_params = eae_decoder.init(subkey, z)

        return e_initial_params, d_initial_params


    def apply_eae(params, x, key):
    
        e_params, d_params = params

        initial_z = jax.nn.sigmoid(eae_encoder.apply(e_params,  x))
        mean = jax.nn.sigmoid(initial_z[:LATENT])
        var = initial_z[LATENT:]
        std = jnp.exp(var * 0.5)
        epsilon = random.normal(key, shape=std.shape)

        z = mean + std * epsilon
        
        out = eae_decoder.apply(d_params, z)

        return out, mean, var
    

    def compute_loss_eae(params, key, x, y, testing=False):
        y_hat, mean, var = apply_eae(params, x, key)

        reconstruction = jnp.abs((y_hat.reshape(y.shape ) - y)).mean()
        kl_divergence = -0.5 * jnp.sum(1 + var -jnp.power(mean, 2) - jnp.exp(var))

        loss = reconstruction + kl_divergence
        if testing: return reconstruction
        return loss
    
    return init_eae, compute_loss_eae   

def process_mlp_batch(key, batch):
    _, x, _ = batch
    x = x.squeeze(0).flatten()
    #noise = random.normal(key, shape=x.shape)
    x_in = x #+ noise
    return x_in, x

def plot_training(mlp_data, emlp_data):
    iters = np.linspace(1, len(mlp_data.losses), num=len(mlp_data.losses))
    epochs = np.linspace(1, len(mlp_data.epoch_loss), num=len(mlp_data.epoch_loss))

    # plot iters losses 
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(iters, mlp_data.losses, label='No equivariance')
    ax.plot(iters, emlp_data.losses, label='Equivariance')
    ax.legend(loc='best')
    fig.savefig('figs/iter_losses.png') 
    plt.close(fig)

    # plot epoch losses
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(epochs, mlp_data.epoch_loss, label='No equivariance')
    ax.plot(epochs, emlp_data.epoch_loss, label='Equivariance') 
    
    ax.plot(epochs, mlp_data.val_losses, label='Validation - No equivariance')
    ax.plot(epochs, emlp_data.val_losses, label='Validation - Equivariance')  
    ax.legend(loc='best')
    fig.savefig('figs/epoch_losses.png') 
    
    plt.close(fig)

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_dcd_dataset('./data/adp-vacuum.pdb', './data/100.dcd', 1, distribution=[100,100])

    key = random.PRNGKey(3121)

    print(10*'-'+' training equivariant ' +'-'*10)

    emlp_optimizer = optax.adam(learning_rate=7e-3)

    batch = next(iter(train_loader))
    x, _ = process_mlp_batch(key, batch)

    init_eae, compute_loss_eae = build_eae()

    initial_params = init_eae(key, x)

    eae_data = fit(key, initial_params, emlp_optimizer, compute_loss_eae, process_mlp_batch, train_loader, EPOCHS, val_loader, test_loader)

    print(10*'-'+' training auto encoder ' +'-'*10)

    ae_optimizer = optax.adam(learning_rate=3e-3)

    batch = next(iter(train_loader))
    x, _ = process_mlp_batch(key, batch)

    ae, compute_loss_ae = build_ae()
    initial_params = ae.init(key, x, key)

    ae_data = fit(key, initial_params, ae_optimizer, compute_loss_ae, process_mlp_batch, train_loader, EPOCHS, val_loader, test_loader)

    plot_training(ae_data, eae_data) 