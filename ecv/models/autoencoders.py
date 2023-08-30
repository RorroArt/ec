import jax 
import jax.numpy as jnp

from ecv.models.encoders import Encoder
from ecv.models.decoders import Decoder

import haiku as hk

from typing import NamedTuple, Callable, Tuple


class AutoEncoder:
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        self._encoder = encoder
        self._decoder = decoder
    @property
    def encoder(self):
        return hk.without_apply_rng(hk.transform(self._encoder))
    
    @property
    def decoder(self):
        return hk.without_apply_rng(hk.transform(self._decoder))

    def init(self, key, input_shape, latent_size): 
        fake_input = jnp.ones(input_shape)
        fake_latent = jnp.ones(latent_size)
        encoder_params =  self.encoder.init(key, fake_input, key)
        decoder_params = self.decoder.init(key, fake_latent)
        return (encoder_params, decoder_params)


    def encode(self, encoder_params:Tuple[hk.Params], inputs, key):
        return self.encoder.apply(encoder_params, inputs, key)
    
    def decode(self, decoder_params:Tuple[hk.Params], latent):
        return self.decoder.apply(decoder_params, latent)

    def apply(self, params, inputs, key):
        encoder_params, decoder_params = params
        z, mean, var = self.encode(encoder_params, inputs, key)
        return self.decode(decoder_params, z), mean, var

    

# Experiment with the simulation run
    # Run the FUNN and the spectral abf on top
    # Restart simulation
