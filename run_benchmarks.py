import jax
import haiku as hk

from emlp.reps import T
from emlp.groups import SO

from ecv.training import Hyperparams
from ecv.models import AutoEncoder
from ecv.models.decoders import SimpleAdjDecoder
from ecv.models.encoders import EGNN, MLP, EMLP
from ecv.benchmarks import SingleTask, SampleEfficiency

# models
N_MOLECULES = 9 # For QM9
HIDDEN = 128
LATENT = 3

egnn = AutoEncoder(
    encoder = EGNN(hidden_nf=int(HIDDEN/2),
        z_dim=LATENT,
        n_layers=1,
        activation=jax.nn.swish,
        reg=1e-3),
    decoder=SimpleAdjDecoder(
        n_nodes=N_MOLECULES,
        hidden=HIDDEN
    )
)
mlp = AutoEncoder(
    encoder = MLP(HIDDEN, LATENT),
    decoder=SimpleAdjDecoder(
        n_nodes=N_MOLECULES,
        hidden=HIDDEN
    )
)
emlp = AutoEncoder(
    encoder = EMLP(N_MOLECULES * T(1), T(1), SO(3), HIDDEN, 1),
    decoder=SimpleAdjDecoder(
        n_nodes=N_MOLECULES,
        hidden=HIDDEN
    )
)

# Global Variables
MODELS = {'Base MLP Encoder': mlp,'Equivariant MLP Encoder': emlp,  'E(n)-GNN Encoder': egnn, }
SAMPLES = [100, 500, 1000, 5000, 10000, 50000,100000]

# Define Benchmarks
SINGLE_HYPERPARAMS = Hyperparams(
    batch_size = 1,
    epochs = 1,
    learning_rate = 1e-3,
    latent_size = LATENT
)


# Initialize Benchmarks
single_task = SingleTask(
    dataset='qm9',
    models = MODELS,
    hyperparams = SINGLE_HYPERPARAMS,
)

sample_task = SampleEfficiency(
    dataset='qm9',
    models = MODELS,
    hyperparams = SINGLE_HYPERPARAMS,
    samples=SAMPLES
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(6352)

    single_task.run(key)
    single_task.save_results('./results')
