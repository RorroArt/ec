import jax
import haiku as hk

from emlp.reps import T
from emlp.groups import SO

from ecv.training import Hyperparams
from ecv.models.decoders import SimpleAdjDecoder
from ecv.models.encoders import EGNN, MLP, EMLP
from ecv.benchmarks import SingleTask, SampleEfficiency

# models
N_MOLECULES = 9 # For QM9
HIDDEN = 128
LATENT = 3


@hk.transform
def egnn(inputs, key):
    model = EGNN(
        hidden_nf=int(HIDDEN/2),
        z_dim=LATENT,
        n_layers=1,
        activation=jax.nn.swish,
        reg=1e-3
    )
    return model(inputs, key)

@hk.transform
def mlp(inputs, key):
    model = MLP(HIDDEN, LATENT)
    return model(inputs, key)

emlp = hk.transform(EMLP(N_MOLECULES * T(1), T(1), SO(3), HIDDEN, 1))

@hk.transform
def simple_decoder(inputs):
    model = SimpleAdjDecoder(
        n_nodes=N_MOLECULES,
        hidden=HIDDEN
    )
    return model(inputs)

# Global Variables
ENCODERS = {'Base MLP Encoder': mlp, 'E(n)-GNN Encoder': egnn, 'Equivariant MLP Encoder': emlp, }
SAMPLES = [100, 500, 1000, 5000, 10000, 50000,100000]

# Define Benchmarks
SINGLE_HYPERPARAMS = Hyperparams(
    batch_size = 1,
    epochs = 100000,
    learning_rate = 1e-3,
    latent_size = LATENT
)


# Initialize Benchmarks
single_task = SingleTask(
    dataset='qm9',
    encoders = ENCODERS,
    decoder = simple_decoder,
    hyperparams = SINGLE_HYPERPARAMS,
)

sample_task = SampleEfficiency(
    dataset='qm9',
    encoders = ENCODERS,
    decoder = simple_decoder,
    hyperparams = SINGLE_HYPERPARAMS,
    samples=SAMPLES
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(6352)

    sample_task.run(key)
    sample_task.save_results('./results')
