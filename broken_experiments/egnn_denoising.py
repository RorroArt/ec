import jax 
import jax.numpy as jnp
import jax.random as random

import numpy as np

import haiku as hk
import optax

import emlp.nn.haiku as ehk

import matplotlib.pyplot as plt

from utils import load_dcd_dataset, bonds_to_graph
from ecv.utils.training import fit
from models import VEGEncoder, SimpleDecoder


def get_bonds_from_pdb(pdb_file_path, bond_distance_threshold=1.6):
    atoms = []
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            record_type = line[0:6].strip()
            if record_type == "ATOM":
                atom_serial = int(line[6:11])
                atom_symbol = line[12:16].strip()
                atom_position = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                atoms.append((atom_serial, atom_symbol, atom_position))

    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = jnp.linalg.norm(jnp.array(atoms[i][2]) - jnp.array(atoms[j][2]))
            if distance <= bond_distance_threshold:
                bonds.append((atoms[i][0] - 1, atoms[j][0] - 1))

    return bonds


pdb_file_path = "data/adp-vacuum.pdb"
bonds = get_bonds_from_pdb(pdb_file_path)

N_MOLECULES = 22
BATCH_SIZE = 1
EPOCHS = 50

BONDS =  jnp.array(bonds)

edges, edge_attr, adj = bonds_to_graph(BONDS, N_MOLECULES)

@hk.transform
def egae(inputs, key):
  encoder = VEGEncoder(
        hidden_nf=32,
        n_layers=1,
        z_dim=3,
        activation=jax.nn.swish,
        reg=1e-3
  )
  decoder = SimpleDecoder( 
    in_ft=N_MOLECULES*3,
    G=64,
  )
  z, mean, var = encoder(inputs, key)
  out = decoder(z)
  return out, mean, var


def process_egae_batch(key, batch):
    _, x, _ = batch
    x = x.squeeze(0)
    x_in = x 
    h = jnp.expand_dims(jnp.ones(x.shape[0]), axis=1)
    return (h, x_in, edges, edge_attr), x

def compute_loss_egae(params, key, x, y, testing=False):
    y_hat, mean, var = egae.apply(params, key, x, key)

    reconstruction = jnp.abs((y_hat.reshape(y.shape ) - y)).mean()
    kl_divergence = -0.5 * jnp.sum(1 + var -jnp.power(mean, 2) - jnp.exp(var))
    loss = reconstruction + kl_divergence
    if testing: return reconstruction
    return loss

def plot_training(mlp_data):
    iters = np.linspace(1, len(mlp_data.losses), num=len(mlp_data.losses))
    epochs = np.linspace(1, len(mlp_data.epoch_loss), num=len(mlp_data.epoch_loss))

    # plot iters losses 
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(iters, mlp_data.losses, label='egnn')
    ax.legend(loc='best')
    fig.savefig('figs/egnn_iter_losses.png') 
    plt.close(fig)
    # plot epoch losses
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(epochs, mlp_data.epoch_loss, label='egnn')
    ax.legend(loc='best')
    fig.savefig('figs/egnn_epoch_losses.png') 
    
    plt.close(fig)



if __name__ == '__main__':
    key = random.PRNGKey(3121)

    train_loader, val_loader, test_loader = load_dcd_dataset('./data/adp-vacuum.pdb', './data/1000.dcd', 1, distribution=[1000,300])

    optimizer = optax.adam(learning_rate=3e-3)

    batch = next(iter(train_loader))
    x, _ = process_egae_batch(key, batch)

    initial_params = egae.init(key, x, key)
        
    training_data = fit(key, initial_params, optimizer, compute_loss_egae, process_egae_batch, train_loader, EPOCHS, val_loader, test_loader)

    plot_training(training_data)