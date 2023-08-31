import jax 
import jax.numpy as jnp
import jax.random as random

import haiku as hk

from ecv.models.layers import EGCL

# Abstrac decoder class
class Decoder(hk.Module):
    pass

class SimpleDecoder_Module(hk.Module):
    def __init__(self, in_ft, hidden):
        super().__init__()
        self.decoder = hk.Sequential([
            hk.Linear(hidden),
            jax.nn.tanh,
            hk.Linear(in_ft),

        ]) 
    def __call__(self, x):
        return self.decoder(x)
    

class SimpleAdjDecoder_Module(hk.Module):
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
    
class SimpleRelDecoder_Module(hk.Module):
    def __init__(self, n_nodes, hidden):
        super().__init__()
        self.decoder = hk.Sequential([
            hk.Linear(hidden),
            jax.nn.tanh,
            hk.Linear(n_nodes),

        ]) 
    def __call__(self, z):
        x_z = self.decoder(z) # expand latent 
        adj = jnp.einsum('i, j -> i j ', x_z, x_z) # dot product

        return adj

# Self explanatory  
class RegressionDecoder_Module(hk.Module):
    def __init__(self, hidden):
        super().__init__()
        self.decoder = hk.Sequential([
            hk.Linear(hidden),
            jax.nn.relu,
            hk.Linear(1),

        ]) 
    def __call__(self, z):
        return self.decoder(z)

# For the basin
class BinaryClassifierDecoder_Module(hk.Module):
    def __init__(self, hidden):
        super().__init__()
        self.decoder = hk.Sequential([
            hk.Linear(hidden),
            jax.nn.relu,
            hk.Linear(1),
            jax.nn.sigmoid
        ]) 
    def __call__(self, z):
        return self.decoder(z)
    
class MultiTaskDecoder_Module(Decoder):
    def __init__(self, n_regression, n_classifiers, hidden):
        super().__init__()
        self.tasks = [RegressionDecoder(hidden) for _ in range(n_regression)]
        self.tasks += [BinaryClassifierDecoder(hidden) for _ in range(n_classifiers)]
    
    def __call__(self, z):
        out = []
        for task in self.tasks:
            out.append(task(z))
        return out

# Write me the wrappers 
def SimpleDecoder(in_ft, hidden):        
    def decoder(z):
        model = SimpleDecoder_Module(in_ft, hidden)
        return model(z)
    return decoder

def SimpleAdjDecoder(n_nodes, hidden):        
    def decoder(z):
        model = SimpleAdjDecoder_Module(n_nodes, hidden)
        return model(z)
    return decoder 

def SimpleRelDecoder(n_nodes, hidden):
    def decoder(z):
        model = SimpleRelDecoder_Module(n_nodes, hidden)
        return model(z)
    return decoder

def RegressionDecoder(hidden):        
    def decoder(z):
        model = RegressionDecoder_Module(hidden)
        return model(z)
    return decoder

def BinaryClassifierDecoder(hidden):
    def decoder(z):
        model = BinaryClassifierDecoder_Module(hidden)
        return model(z)
    return decoder

def MultiTaskDecoder(n_regression, n_classifiers, hidden):
    def decoder(z):
        model = MultiTaskDecoder_Module(n_regression, n_classifiers, hidden)
        return model(z)
    return decoder


        
        