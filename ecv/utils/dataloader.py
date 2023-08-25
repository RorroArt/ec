import jax
import jax.numpy as jnp
from jax.random import permutation, PRNGKey


class DataLoader:
    def __init__(self, data, batch_size, edges=None):
        self.indeces, self.node_features, self.coords = data
        self.batch_size = batch_size
        self.edges = edges
        self.idx = 0
        self.seed = 1
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < len(self.indeces) - self.batch_size:
            self.idx += self.batch_size
            batch_idx = self.indeces[self.idx - self.batch_size: self.idx]
            if self.edges is not None:
                return self.node_features[batch_idx], self.coords[batch_idx], self.edges[batch_idx]
            else:
                return self.node_features[batch_idx], self.coords[batch_idx],   
        else:
            self.data = permutation(PRNGKey(self.seed), self.indeces)
            self.idx = 0
            self.seed += 1
            raise StopIteration 
    
    def __len__(self):
        return len(self.indeces)
