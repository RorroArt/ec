import jax 
import jax.numpy as jnp
import jax.random as random

import haiku as hk

import emlp.nn.haiku as ehk

from ecv.models.layers import EGCL

# Abstract Encoder class
class Encoder(hk.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
    
    def reparametrize(self, key, latent):
        mean = latent[:self.z_dim]; var = latent[self.z_dim:]
        std = jnp.exp(var * 0.5)
        epsilon = random.normal(key, shape=std.shape) 
        z = mean + std*epsilon
        return z, mean, var
    
    def encode(self, x): pass
    
    def __call__ (self, x, key):
        latent = self.encode(x)
        return self.reparametrize(key, latent)


class MLP_Module(Encoder):
    def __init__(self, hidden, z_dim):
        super().__init__(z_dim=z_dim)
        self.encode = hk.Sequential([
            hk.Linear(hidden),
            jax.nn.tanh,
            hk.Linear(z_dim * 2),
        ])

class EGNN_Module(Encoder):
    def __init__(
        self,
        hidden_nf,
        z_dim,
        n_layers,
        activation,
        reg,
        normalize=True,
    ):
        super().__init__(z_dim=z_dim)
        self.reg = reg
        self.n_layers = n_layers

        self.embedding_in = hk.Linear(hidden_nf)

        self.embedding_out_x = hk.Sequential([jax.nn.tanh, hk.Linear(z_dim * 2)])
        
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(EGCL(hidden_nf, hidden_nf, activation, normalize))

    def encode(self, inputs, train=True):
        h, x, edges, edge_attr = inputs

        h = self.embedding_in(h)

        for i in range(self.n_layers):
            h, x = self.layers[i](h, edges, x, edge_attr)
            x -= x * self.reg
        
        return self.embedding_out_x(x.flatten())

# Wrappers to match the pytorch API  

def MLP(hidden, z_dim):
    def encoder(inputs, key):
        model = MLP_Module(hidden, z_dim)
        return model(inputs, key)
    
    return encoder

def EGNN(hidden_nf, z_dim, n_layers, activation, reg):
    def encoder(inputs, key):
        model = EGNN_Module(
            hidden_nf=hidden_nf,
            z_dim=z_dim,
            n_layers=n_layers,
            activation=jax.nn.swish,
            reg=1e-3,
        )
        return model(inputs, key)
    return encoder

from emlp.reps import Rep
from emlp.nn.haiku import uniform_rep, EMLPBlock, Linear, Sequential

# Most of this code was taken from https://github.com/mfinzi/equivariant-MLP/blob/master/emlp/nn/haiku.py
# Is customized to work as an encoder



def EMLP(rep_in,rep_out,group,ch=384,num_layers=3):
    rep_in =rep_in(group)
    rep_out = rep_out * 2; rep_out = rep_out(group)
    # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
    if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]
    elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
    else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
    # assert all((not rep.G is None) for rep in middle_layers[0].reps)
    reps = [rep_in]+middle_layers
    # logging.info(f"Reps: {reps}")
    def encoder(inputs, key):
        model = Encoder(z_dim=int(rep_out.size()/2))
        model.encode = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],rep_out)
        )
        return model(inputs, key)
    return encoder