import jax 
import jax.numpy as jnp

from ecv.datasets import load_qm9 
from ecv.training import fit
from ecv.utils import bonds_to_graph

import optax

import json


DATASETS = {
    'qm9': load_qm9
}

class Benchmark:
    def __init__(self, dataset, models, hyperparams):
        self.train_loader, self.val_loader, self.test_loader = DATASETS[dataset]()
        self.models = models
        self.hyperparams = hyperparams
        self.results = []

    def run(self, key): pass

    def save_results(self, file_path): 
        if len( self.results ) != 0:
            for label, data in self.results:
                data_dict = data._asdict()
                with open(file_path + f'/{label}', 'w') as json_file:
                    json.dump(data_dict, json_file, indent=4)
        self.results = []

class SingleTask(Benchmark):
    def __init__(self, dataset, models, hyperparams):
        super().__init__(dataset=dataset, models=models, hyperparams=hyperparams)

    def run(self, key):
        fake_batch = next(iter(self.train_loader))
        fake_latent = jnp.ones(self.hyperparams.latent_size)
        for name, model in self.models.items():
            print(f'\n Training {name} \n')
            if name == 'E(n)-GNN Encoder':
                def preprocess_batch(key, batch): 
                    h, x, bonds = batch
                    x = x.squeeze(0)
                    edges, edge_attr, adj = bonds_to_graph(jnp.array(bonds[0]), x.shape[0])
                    return (h, x, edges, edge_attr), adj
            else:
                def preprocess_batch(key, batch): 
                    _, x, bonds = batch
                    _, _, adj = bonds_to_graph(jnp.array(bonds[0]), x.shape[1])
                    x = x.squeeze(0).flatten()
                    return x, adj

            def loss_fn(params, key, x, y, testing=False): 
                y_hat, mean, var = model.apply(params, x, key)

                reconstruction = optax.sigmoid_binary_cross_entropy(y_hat, y).mean()
                kl_divergence = -0.5 * jnp.sum(1 + var -jnp.power(mean, 2) - jnp.exp(var))

                loss = reconstruction + kl_divergence
                if testing: return reconstruction
                return loss
            optimizer = optax.adam(self.hyperparams.learning_rate)
            fake_input, _  = preprocess_batch(key, fake_batch)
            params = model.init(key, fake_input.shape,  self.hyperparams.latent_size)
            training_data, params = fit(key, params, optimizer,loss_fn, preprocess_batch, self.train_loader, self.hyperparams.epochs, self.val_loader, self.test_loader )
            self.results.append((f'Single task  - {name}' ,training_data))


class SampleEfficiency(Benchmark):
    def __init__(self, dataset, models, hyperparams, samples):
        super().__init__(dataset=dataset, models=models, hyperparams=hyperparams)
        self.samples = samples

    def run(self, key):
        fake_batch = next(iter(self.train_loader))
        fake_latent = jnp.ones(self.hyperparams.latent_size)
        for sample in self.samples:
            print(f'\n Sample size {sample}\n')

            for name, model in self.models.items():
                print(f'\n Training {name} \n')
                if name == 'E(n)-GNN Encoder':
                    def preprocess_batch(key, batch): 
                        h, x, bonds = batch
                        x = x.squeeze(0)
                        edges, edge_attr, adj = bonds_to_graph(jnp.array(bonds[0]), x.shape[0])
                        return (h, x, edges, edge_attr), adj
                else:
                    def preprocess_batch(key, batch): 
                        _, x, bonds = batch
                        _, _, adj = bonds_to_graph(jnp.array(bonds[0]), x.shape[1])
                        x = x.squeeze(0).flatten()
                        return x, adj

                def loss_fn(params, key, x, y, testing=False): 
                    y_hat, mean, var = model.apply(params, x, key)
                    
                    reconstruction = optax.sigmoid_binary_cross_entropy(y_hat, y).mean()
                    kl_divergence = -0.5 * jnp.sum(1 + var -jnp.power(mean, 2) - jnp.exp(var))

                    loss = reconstruction + kl_divergence
                    if testing: return reconstruction
                    return loss
                optimizer = optax.adam(self.hyperparams.learning_rate)
                fake_input, _  = preprocess_batch(key, fake_batch)
                params = model.init(key, fake_input.shape,  self.hyperparams.latent_size)
                training_data, params = fit(key, params, optimizer,loss_fn, preprocess_batch, self.train_loader, int(self.hyperparams.epochs / sample), self.val_loader, self.test_loader, max_steps=sample, val_max_steps=1000, test_max_steps=1000 )
                self.results.append((f'Sample Eff - Sample {sample} - {name}' ,training_data))
            self.save_results('./results')

