import jax 
import jax.numpy as jnp
import jax.random as random

import haiku as hk

from ecv.models.layers import EGCL

class Decoder(hk.Module):
    pass

class SimpleDecoder(hk.Module):
    def __init__(self, in_ft, hidden):
        super().__init__()
        self.decoder = hk.Sequential([
            hk.Linear(G),
            jax.nn.tanh,
            hk.Linear(hidden),

        ]) 
    def __call__(self, x):
        return self.decoder(x)
    

class SimpleAdjDecoder(hk.Module):
    def __init__(self, n_nodes, hidden):
        super().__init__()
        self.decoder = hk.Sequential([
            hk.Linear(hidden),
            jax.nn.tanh,
            hk.Linear(n_nodes),

        ]) 
    def __call__(self, z):
        x_z = self.decoder(z) # expand latent 
        adj_weights = jnp.einsum('i, j -> i j ', x_z, x_z) # dot product
        adj = jax.nn.sigmoid(adj_weights) # compute adjacency matrix
        
        return adj

# Self explanatory  
class PotentialEnergyDecoder(hk.Module):
    pass

# For the basin
class BinaryClassifierDecoder(hk.Module):
    pass