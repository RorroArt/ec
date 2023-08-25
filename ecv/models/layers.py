import jax 
import jax.numpy as jnp
import jax.random as random

import haiku as hk

from einops import rearrange

"""
E(n) Equivariant Graph Convolutional layers. Taken from: https://arxiv.org/pdf/2102.09844.pdf
"""

def coord2radial(edge_index, coord, normalize=True, epsilon=1e-8):
    row, col = edge_index
    diff = coord[row] - coord[col]
    rad = (diff**2).sum(-1)
    rad = rearrange(rad, 'i -> i ()')

    if normalize:
        norm= jnp.sqrt(jax.lax.stop_gradient(rad)) + epsilon
        diff = diff / norm

    return rad, diff

class EGCL(hk.Module):
    """
    E(n) Equivariant Graph Convolutional networks with momentum.
    """
    def __init__(
        self,
        out_nf,
        hidden_nf,
        activation,
        normalize=True,
    ):
        super().__init__()
        self.edge_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf),
            activation
        ])

        self.node_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf)
        ])
        self.coord_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(1)
        ])
        self.out_nf = out_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.normalize = normalize

    def __call__(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        r, c = edge_index
        rad, diff = coord2radial(edge_index, coord)
        
        if edge_attr is not None:
            edge_input = jnp.concatenate([h[r], h[c], rad, edge_attr], axis=1)
        else:
            edge_input = jnp.concatenate([h[r], h[c], rad], axis=1)
        m_ij = self.edge_op(edge_input)

        weights = self.coord_op(m_ij)
        coors_sum = jax.ops.segment_sum(diff * weights, r, num_segments=coord.shape[0])
        coord = coord + coors_sum

        m_i = jax.ops.segment_sum(m_ij, r, num_segments=h.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([h, m_i, node_attr], 1)
        else: 
            agg = jnp.concatenate([h, m_i], 1)
        h_out = h + self.node_op(agg)

        return h_out, coord


class EGCL_v(hk.Module):
    """
    E(n) Equivariant Graph Convolutional networks with momentum (accounts for input velocity)
    """
    def __init__(
        self,
        out_nf,
        hidden_nf,
        activation,
        normalize=True,
    ):
        super().__init__()
        self.edge_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf),
            activation
        ])

        self.node_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf)
        ])
        self.coord_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(1)
        ])
        self.vel_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(1)
        ])

        self.out_nf = out_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.normalize = normalize

    def __call__(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        r, c = edge_index
        rad, diff = coord2radial(edge_index, coord)
        
        if edge_attr is not None:
            edge_input = jnp.concatenate([h[r], h[c], rad, edge_attr], axis=1)
        else:
            edge_input = jnp.concatenate([h[r], h[c], rad], axis=1)
        m_ij = self.edge_op(edge_input)

        weights = self.coord_op(m_ij)
        coors_sum = jax.ops.segment_sum(diff * weights, r, num_segments=coord.shape[0])
        vel_weights = self.vel_op(h)
        coord = coord + coors_sum + vel * vel_weights

        m_i = jax.ops.segment_sum(m_ij, r, num_segments=h.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([h, m_i, node_attr], 1)
        else: 
            agg = jnp.concatenate([h, m_i], 1)
        h_out = self.node_op(agg)

        return h_out, coord
    
class EGCL_prop_v(hk.Module):
    """
    E(n) Equivariant Graph Convolutional networks with momentum (accounts for input velocity).
    It propagates the updated velocity to the next layer
    """
    def __init__(
        self,
        out_nf,
        hidden_nf,
        activation,
        normalize=True,
    ):
        super().__init__()
        self.edge_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf),
            activation
        ])

        self.node_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf)
        ])
        self.coord_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(1)
        ])
        self.vel_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(1)
        ])
        self.out_nf = out_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.normalize = normalize

    def __call__(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
        r, c = edge_index
        rad, diff = coord2radial(edge_index, coord)
        
        if edge_attr is not None:
            edge_input = jnp.concatenate([h[r], h[c], rad, edge_attr], axis=1)
        else:
            edge_input = jnp.concatenate([h[r], h[c], rad], axis=1)
        m_ij = self.edge_op(edge_input)

        weights = self.coord_op(m_ij)
        coors_sum = jax.ops.segment_sum(diff * weights, r, num_segments=coord.shape[0])
        vel_weights = self.vel_op(h)
        vel = vel * vel_weights 
        coord = coord + vel

        m_i = jax.ops.segment_sum(m_ij, r, num_segments=h.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([h, m_i, node_attr], 1)
        else: 
            agg = jnp.concatenate([h, m_i], 1)
        h_out = self.node_op(agg)

        return h_out, coord, vel