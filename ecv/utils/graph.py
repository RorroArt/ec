import jax 
import jax.numpy as jnp

from jax.experimental import sparse

def adj_to_edge(adj):
    n_nodes = adj.shape[0]
    edge_attr = jnp.zeros((n_nodes ** 2 - n_nodes, 1))
    rows, cols = [], []
    k = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
                edge_attr = edge_attr.at[k, 0].set(adj[i, j])
                k += 1
    edges = [jnp.array(rows), jnp.array(cols)]
    return edges, edge_attr

def bonds_to_graph(bonds, n_nodes):
    bonds = bonds.astype(jnp.int32)
    v = jnp.ones(bonds.shape[0])
    adj = sparse.BCOO((v,bonds), shape=(n_nodes, n_nodes)).todense()

    edges, edge_attr = adj_to_edge(adj)
    return edges, edge_attr, adj


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