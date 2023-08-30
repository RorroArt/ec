import os 
import json

from ecv.utils import DataLoader, untar_file, download_file

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import rdmolops
from tqdm import tqdm 

import jax
import jax.numpy as jnp
import numpy as np

from jax.random import permutation, PRNGKey

# Molecules dataset for bechmark purposes

# QM9
QM9_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
QM9_DESTINATION_TAR = './data/qm9.tar.gz'
QM9_DESTINATION_FOLDER = './data/qm9'

def load_qm9(url = QM9_URL, tar_destination=QM9_DESTINATION_TAR, destination_folder = QM9_DESTINATION_FOLDER, batch_size=1, seed=43):
    
    # Create the necessesary files and donwload the dataset
    destination_dir = os.path.dirname(tar_destination)
    tar_exists = os.path.exists(tar_destination )

    sdf_path = destination_folder + '/gdb9.sdf'
    sdf_dir = os.path.dirname(sdf_path)
    sdf_exists = os.path.exists(sdf_path)

    processed_path = destination_folder + '/qm9.json'
    processed_dir = os.path.dirname(processed_path)
    processed_exists = os.path.exists(processed_path)

    if not tar_exists and not sdf_exists:
        if not os.path.exists(destination_dir): os.makedirs(destination_dir)
        download_file(url, tar_destination) 
    
    if not sdf_exists:
        if not os.path.exists(sdf_dir ): os.makedirs(sdf_dir)
        untar_file(tar_destination, destination_folder)

    if not processed_exists:
        supplier = Chem.SDMolSupplier(sdf_path)

        max_atoms = 9
        all_coordinates = []
        all_bonds = []
        # Use tqdm to create a progress bar for the iterations
        for mol in tqdm(supplier, desc="Processing molecules"):
            if mol is not None and mol.GetNumAtoms() > 1:
                # Retrieve the first conformation (molecule's 3D structure)
                conformer = mol.GetConformer()

                # Extract atom coordinates and pad smaller molecules
                num_atoms = mol.GetNumAtoms()
                if num_atoms < max_atoms:
                    padding = max_atoms - num_atoms
                    pad_coordinates = jnp.zeros((padding, 3), dtype=jnp.float32)
                    coordinates = jnp.array([conformer.GetAtomPosition(i) for i in range(num_atoms)])
                    padded_coordinates = jnp.concatenate((coordinates, pad_coordinates))
                else:
                    coordinates = jnp.array([conformer.GetAtomPosition(i) for i in range(max_atoms)])
                    padded_coordinates = coordinates

                all_coordinates.append(padded_coordinates)

                # Calculate the edges (connections) between atoms
                bonds = []
                for bond in mol.GetBonds():
                    bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                all_bonds.append(bonds)

        # Convert to JAX arrays
        coords = jnp.array(all_coordinates)
        all_bonds = np.array(all_bonds, dtype=object)

        data_dict = {'Coordinates': coords.tolist(), 'Bonds': all_bonds.tolist()}
        with open(processed_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)

        with open(csv_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            return [row['homo'] for row in reader]

    else:
        with open(processed_path, 'r') as json_file:
            data_dict = json.load(json_file)

        coords = jnp.array(data_dict['Coordinates'])
        all_bonds = np.array(data_dict['Bonds'], dtype=object)


    # Split dataset
    n_mols = coords.shape[0]
    indeces = permutation(PRNGKey(seed), jnp.linspace(0, n_mols-1, num=n_mols))
    train_idx = int(n_mols * 0.8)
    val_idx = train_idx + int(n_mols * 0.1)
    
    fake_node_features = jnp.ones(shape=(n_mols, 1)) # GNNs are going to need this 

    train_selection = indeces[:train_idx].astype(jnp.int32)
    train_indices = jnp.linspace(0, train_idx - 1, num=train_idx, dtype=jnp.int32)
    train_data = (train_indices, fake_node_features[train_selection], coords[train_selection])
    train_loader = DataLoader(train_data, batch_size=batch_size, edges=all_bonds[train_selection])
        
    val_selection = indeces[train_idx:val_idx].astype(jnp.int32)
    val_indices = jnp.linspace(0, (val_idx-train_idx) - 1, num=(val_idx - train_idx), dtype=jnp.int32)
    val_data = (val_indices, fake_node_features[val_selection], coords[val_selection])
    val_loader = DataLoader(val_data, batch_size=batch_size, edges=all_bonds[val_selection])
    
    test_selection = indeces[val_idx:].astype(jnp.int32)
    test_indices = jnp.linspace(0, (n_mols- val_idx) - 1, num=(n_mols - val_idx), dtype=jnp.int32) 
    test_data = (test_indices, fake_node_features[test_selection], coords[test_selection])
    test_loader = DataLoader(test_data, batch_size=batch_size, edges=all_bonds[test_selection])

    return train_loader, val_loader, test_loader



