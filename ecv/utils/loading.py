import requests
import gzip
import tarfile

import gsd.hoomd as gsd_hoomd
import MDAnalysis as mda

import jax 
import jax.numpy as jnp
from jax.random import permutation, PRNGKey

from ecv.utils.dataloader import DataLoader

def download_file(url, destination):
    try:
        response = requests.get(url)
        with open(destination, "wb") as f:
            f.write(response.content)
        print("File downloaded successfully.")
    except Exception as e:
        print('Error when downloading file', e)
        pass


def untar_file(tar_path, destiantion_folder):
  try:
    with gzip.open(tar_path, 'rb') as gz:
        with tarfile.open(fileobj=gz, mode='r') as tar:
            tar.extractall(path=destiantion_folder)
  except Exception as e:
    print('Error when decompressing file', e)
    pass
  

def load_gsd_dataset(
        file_path,
        batch_size, 
        distribution=None,
        seed=1,
        properties=['mass']
        ):
    """
    Load dataset from a gsd hoomd simulation file
    
    Args:
        file_path: Path and name of the file that will be loaded.
        batch_size: Size of the batch.
        ditribution: Distribution between Training, Validation, and Testing datasets.
        seed: Seed for shuffleing the dataset.
        properties: Properties of the particle that will be loaded 
            'mass', 'diameter', etc.

    Returns:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Testing data loader.

    """

    trajectory = gsd_hoomd.open(file_path, 'rb')
    
    indeces = []
    node_features = []
    coords = []
    velocities = []
    
    for i, frame in enumerate(trajectory):
        particles = frame.particles 
        indeces.append(i)
        node_features.append([getattr(particles, feature) for feature in properties])
        coords.append(particles.position)
        velocities.append(particles.velocity)
    trajectory.close()

    indeces = permutation(PRNGKey(seed), jnp.array(indeces))
    node_features = jnp.transpose(jnp.array(node_features), (0,2,1))
    node_features = jax.nn.standardize(node_features,axis=1)
    data = (indeces, node_features, jnp.array(coords), jnp.array(velocities))

    if distribution is not None:

        batches = jnp.floor(data.shape[0] / batch_size)
        train_idx = batches*distribution[0]
        val_idx = train_idx + batches*distribution[1]
        
        train_loader = DataLoader(data[:train_idx], batch_size=batch_size)
        val_loader = DataLoader(data[train_idx:val_idx], batch_size=batch_size)
        test_loader = DataLoader(data[val_idx:], batch_size=batch_size)

        return train_loader, val_loader, test_loader
    
    return DataLoader(data, batch_size=batch_size)

def load_pdb_dataset(
        file_path,
        batch_size, 
        distribution=None,
        seed=1,
        properties=['mass']
        ):

    trajectory = mda.Universe(file_path)
    
    indeces = []
    node_features = []
    coords = []

    for i, frame in enumerate(trajectory):
        particles = frame.atoms 
        indeces.append(i)
        node_features.append([getattr(particles, feature) for feature in properties])
        coords.append(particles.position)
    trajectory.close()

    indeces = permutation(PRNGKey(seed), jnp.array(indeces))
    node_features = jnp.transpose(jnp.array(node_features), (0,2,1))
    node_features = jax.nn.standardize(node_features,axis=1)
    coords = jnp.array(coords)
    data = (indeces, node_features, coords, jnp.zeros(coords.shape))

    if distribution is not None:

        batches = jnp.floor(len(indeces) / batch_size)
        train_idx = batches*distribution[0]
        val_idx = train_idx + batches*distribution[1]
        
        train_loader = DataLoader(data[:train_idx], batch_size=batch_size)
        val_loader = DataLoader(data[train_idx:val_idx], batch_size=batch_size)
        test_loader = DataLoader(data[val_idx:], batch_size=batch_size)

        return train_loader, val_loader, test_loader
    
    return DataLoader(data, batch_size=batch_size)

def load_dcd_dataset(
        structure_file_path,
        traj_file_path,
        batch_size, 
        distribution=None,
        seed=1,
        properties=['masses']
        ):

    universe = mda.Universe(structure_file_path, traj_file_path)
    trajectory = universe.trajectory
    
    indeces = []
    node_features = []
    coords = []

    for i, frame in enumerate(trajectory):
        particles = universe.atoms 
        indeces.append(i)
        node_features.append([getattr(particles, feature) for feature in properties])
        coords.append(frame.positions)
    trajectory.close()

    indeces = permutation(PRNGKey(seed), jnp.array(indeces))
    node_features = jnp.transpose(jnp.array(node_features), (0,2,1))
    node_features = jax.nn.standardize(node_features,axis=1)
    coords = jnp.array(coords)
    data = (indeces, node_features, coords, jnp.zeros(coords.shape))

    if distribution is not None:

        train_idx = int(distribution[0])
        val_idx = train_idx + int(distribution[1])

        train_indices = indeces[:train_idx]
        train_data = (train_indices, node_features[train_indices], coords[train_indices], jnp.zeros(coords.shape)[train_indices])
        train_loader = DataLoader(train_data, batch_size=batch_size)
        
        val_indices = indeces[train_idx:val_idx]
        val_data = (val_indices, node_features[val_indices], coords[val_indices], jnp.zeros(coords.shape)[val_indices])
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        test_indices = indeces[val_idx:]
        test_data = (indeces[test_indices], node_features[test_indices], coords[test_indices], jnp.zeros(coords.shape)[test_indices])
        test_loader = DataLoader(test_data, batch_size=batch_size)

        return train_loader, val_loader, test_loader
    
    return DataLoader(data, batch_size=batch_size)