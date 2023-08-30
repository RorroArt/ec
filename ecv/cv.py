import jax
import jax.numpy as jnp

from models import Model # TODO: Implement this

import haiku as hk
import obta

from typing import NamedTuple

import pysages
from pysages.methods import HistogramLogger, HarmonicBias

class CVLearnerState(NamedTuple):
    simulation: pysages.SimulationContext
    method: pysages.BiasingMethod
    model: callable
    optimizer: jax.experimental.optimizers.Optimizer

class CVLearner:
    def __init__(
        self,
        simulation_context, 
        harmonic_cv, 
        k,
        harmonic_centers,
        method,
        model,
        optimizer,
        loss_fn,
    ):
        self.simulation = simulation_context
        
        
        self.method = method
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.harmonic_bias = method = (HarmonicBias(harmonic_cv, k, harmonic_center)

    # Step 1: Generate initial data
    def generate_data(self, steps: 2000): # Generates dataset of size steps * 2
        steps_b1, steps_b2 = steps # Steps per basin

        self.callback = HistogramLogger(100)
        run_result = pysages.run(self.harmonic_bias, self.simulation_context, self.int(steps_b1), self.callback, profile=True)

    # Step 2: Train the model
    def train_model(self):
        pass
        
    # Step 3