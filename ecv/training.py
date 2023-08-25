import jax 
import jax.numpy as jnp

from jax import random

import haiku as hk
import optax

import time
from typing import NamedTuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

class Hyperparams(NamedTuple):
  batch_size: int
  epochs: int
  learning_rate: float
  latent_size: int 

class TrainingData(NamedTuple):
  losses: list
  epoch_loss: list
  val_losses: Optional[list]
  test_loss: Optional[float]

def build_update_function(optimizer, loss_fn):
  @jax.jit
  def update(params, key, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, key, x, y)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss
  return update

def fit(
    key,
    params,
    optimizer,
    loss_fn,
    process_batch, 
    data_loader, 
    epochs,
    validation_loader=None,
    test_loader=None,
    max_steps=None,
    val_max_steps=None,
    test_max_steps=None):
  state = optimizer.init(params)
  update = build_update_function(optimizer, loss_fn)
  total_losses = []
  all_losses = []
  val_losses = [] if validation_loader is not None else None
  test_loss = None

  # Training loop
  for i in range(epochs):
    losses = []
    start = time.time()
    loss = 0
    max_steps = len(data_loader) if max_steps is None else max_steps
    bar = tqdm(enumerate(data_loader), total=max_steps)
    for j, batch in bar:
      key, subkey = random.split(key)
      x, y = process_batch(key, batch)
      params, state, loss = update(params, key, state, x, y)
      losses.append(loss.item())
      all_losses.append(loss.item())
      bar.set_description(f'Epoch {i}, Loss: {loss: .3f}')
      if j == max_steps: break
    end = time.time()
    total_loss = jnp.array(losses).mean()
    total_losses.append(total_loss.item())

    # Validate at the end of each epoch
    if validation_loader is not None: 
      val_loss = []
      val_max_steps = len(validation_loader) if max_steps is None else val_max_steps 
      for j, batch in tqdm(enumerate(validation_loader), 'Validating', total=val_max_steps):
        key, subkey = random.split(key)
        x, y = process_batch(key, batch)
        loss = loss_fn(params, key, x, y) 
        val_loss.append(loss.item())
        if j == val_max_steps: break
      val_loss = jnp.array(val_loss).mean().item()
      val_losses.append(val_loss)
      print(f'Epoch: {i} - loss: {total_loss: .3f} - Validation loss - {val_loss: .3f} - Epoch Execution time: {(end-start): .3f} sec \n')
    else:
      print(f'Epoch: {i} - loss: {total_loss: .3f} - Execution time: {(end-start): .3f} sec \n')

  # Test at the end of training
  if test_loader is not None:
      test_loss = []
      test_max_steps = len(validation_loader) if max_steps is None else test_max_steps 
      for j, batch in tqdm(enumerate(test_loader), 'Testing', total=test_max_steps):
        key, subkey = random.split(key)
        x, y = process_batch(key, batch)
        loss = loss_fn(params, key, x, y, True) 
        test_loss.append(loss.item())
        if j == test_max_steps: break 
      test_loss = jnp.array(val_loss).mean().item()
      print(f'Training finished! - test loss: {(test_loss):.3f} \n')
    
  return TrainingData(all_losses, total_losses, val_losses, test_loss), params
