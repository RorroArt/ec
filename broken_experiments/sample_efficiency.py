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
HIDDEN = 512
LATENT = 3
STEPS = 30000

BONDS =  jnp.array(bonds)

edges, edge_attr, adj = bonds_to_graph(BONDS, N_MOLECULES)

class AE(hk.Module):
    def __init__(self):
        super().__init__()
        self.encoder = hk.Sequential([
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(HIDDEN),
            jax.nn.tanh,
            hk.Linear(LATENT * 2),
        ]) 
        self.decoder = hk.Sequential([
            hk.Linear(HIDDEN),
            jax.nn.tanh,
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

def build_egae():
    @hk.transform
    def egae(inputs, key):
        encoder = VEGEncoder(
                hidden_nf=int(HIDDEN / 2),
                n_layers=2,
                z_dim=3,
                activation=jax.nn.swish,
                reg=1e-3
        )
        decoder = SimpleDecoder( 
            in_ft=N_MOLECULES*3,
            G=HIDDEN,
        )
        z, mean, var = encoder(inputs, key)
        out = decoder(z)
        return out, mean, var

    def compute_loss_egae(params, key, x, y, testing=False):
        y_hat, mean, var = egae.apply(params, key, x, key)

        reconstruction = jnp.abs((y_hat.reshape(y.shape ) - y)).mean()
        kl_divergence = -0.5 * jnp.sum(1 + var -jnp.power(mean, 2) - jnp.exp(var))
        loss = reconstruction + kl_divergence
        if testing: return reconstruction
        return loss
    
    return egae, compute_loss_egae

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
    eae_encoder = hk.without_apply_rng(hk.transform(ehk.EMLP(N_MOLECULES * V,2*T(1), SO(3), ch=HIDDEN, num_layers=2)))

    @hk.without_apply_rng
    @hk.transform
    def eae_decoder(inputs):
        model = SimpleDecoder(in_ft=N_MOLECULES*3,G=HIDDEN,)
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
        mean = initial_z[:LATENT]
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

def process_egae_batch(key, batch):
    _, x, _ = batch
    x = x.squeeze(0)
    x_in = x 
    h = jnp.expand_dims(jnp.ones(x.shape[0]), axis=1)
    return (h, x_in, edges, edge_attr), x

def plot_training(mlp, emlp, egnn, i):
    iters = np.linspace(1, len(mlp.losses), num=len(mlp.losses))
    epochs = np.linspace(1, len(mlp.epoch_loss), num=len(mlp.epoch_loss))

    # plot iters losses 
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(np.linspace(1, len(mlp.losses), num=len(mlp.losses)), mlp.losses, label='MLP')
    ax.plot(np.linspace(1, len(emlp.losses), num=len(emlp.losses)), emlp.losses, label='SO(3)-MLP')
    ax.plot(np.linspace(1, len(egnn.losses), num=len(egnn.losses)), egnn.losses, label='E(3)-GNN')
    ax.legend(loc='best')
    fig.savefig(f'figs/{i}_iter_losses.png') 
    plt.close(fig)

    # plot epoch losses
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(np.linspace(1, len(mlp.epoch_loss), num=len(mlp.epoch_loss)), mlp.epoch_loss, label='MLP')
    ax.plot(np.linspace(1, len(emlp.epoch_loss), num=len(emlp.epoch_loss)), emlp.epoch_loss, label='SO(3)-MLP')
    ax.plot(np.linspace(1, len(egnn.epoch_loss), num=len(egnn.epoch_loss)), egnn.epoch_loss, label='E(3)-GNN')  
    
    ax.plot(np.linspace(1, len(mlp.val_losses), num=len(mlp.val_losses)), mlp.val_losses, label='MLP - Val')
    ax.plot(np.linspace(1, len(emlp.val_losses), num=len(emlp.val_losses)), emlp.val_losses, label='SO(3)-MLP - Val')
    ax.plot(np.linspace(1, len(egnn.val_losses), num=len(egnn.val_losses)), egnn.val_losses, label='E(3)-GNN - Val')
    
    ax.legend(loc='best')
    fig.savefig(f'figs/{i}_epoch_losses.png') 

def plot_efficiency(mlp, emlp, egnn, samples):
    # plot iters losses 
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.plot(samples, mlp, label='MLP')
    ax.plot(samples, emlp, label='SO(3)-MLP')
    ax.plot(samples, egnn, label='E(3)-GNN')
    ax.legend(loc='best')
    fig.savefig(f'figs/efficiency_plot.png') 
    plt.close(fig)

if __name__ == '__main__':
    samples = [100,500, 1000, 5000]; val_samples = [100, 100, 300, 300]
    mlp_test = []
    emlp_test = []
    egnn_test = []
    
    key = random.PRNGKey(632)

    for i, j in zip(samples, val_samples):
        EPOCHS = int(STEPS / i)
        print(30*'-'+f' training with {i} samples ' +'-'*30)

        train_loader, val_loader, test_loader = load_dcd_dataset('./data/adp-vacuum.pdb', f'./data/{i}.dcd', 1, distribution=[i,j])

        print(10*'-'+' training egnn ' +'-'*10)


        optimizer = optax.adam(learning_rate=1e-4)
        egae, compute_loss_egae = build_egae()
        
        batch = next(iter(train_loader))
        x, _ = process_egae_batch(key, batch)

        initial_params = egae.init(key, x, key)
            
        egnn_data = fit(key, initial_params, optimizer, compute_loss_egae, process_egae_batch, train_loader, EPOCHS, val_loader, test_loader)


        print(10*'-'+' training mlp ' +'-'*10)

        ae_optimizer = optax.adam(learning_rate=1e-4)

        
        x, _ = process_mlp_batch(key, batch)

        ae, compute_loss_ae = build_ae()
        initial_params = ae.init(key, x, key)

        ae_data = fit(key, initial_params, ae_optimizer, compute_loss_ae, process_mlp_batch, train_loader, EPOCHS, val_loader, test_loader)
        print(10*'-'+' training emlp ' +'-'*10)

        emlp_optimizer = optax.adam(learning_rate=1e-4)

        x, _ = process_mlp_batch(key, batch)

        init_eae, compute_loss_eae = build_eae()

        initial_params = init_eae(key, x)

        eae_data = fit(key, initial_params, emlp_optimizer, compute_loss_eae, process_mlp_batch, train_loader, EPOCHS, val_loader, test_loader)
        
        plot_training(ae_data, eae_data, egnn_data, i) 

        mlp_test.append(ae_data.test_loss); emlp_test.append(eae_data.test_loss); egnn_test.append(egnn_data.test_loss)

    plot_efficiency(mlp_test, emlp_test, egnn_test, samples)