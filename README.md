# Equivariant Collective Variables

Do you need a collective variable? Use me to learn one (I need a better and more serious introduction here)

## Instalation 

1. Create a new conda environment using python 3.8.

You would need to have some flavor of conda installed in your system.

2. Install the deep learning dependencies.

- For jax follow the instructions in the [official website](https://jax.readthedocs.io/en/latest/#installation) and make sure to add the correct gpu support for your system.
- Install haiku, optax and einsum.

`
pip install git+https://github.com/deepmind/dm-haiku optax einsum
`

3. Install your prefered backend

- For OpenMM you can installing using conda as specified in the main website: 
    - After that, install the OpenMM-dlext plugging following this instructions in this [repo](https://github.com/SSAGESLabs/openmm-dlext#readme).

- For Hoomd you wnat to ensure GPU support you will have to build from the source (We currently only support version 2.9.7): [instructions](https://hoomd-blue.readthedocs.io/en/v2.9.7/installation.html#compiling-from-source)
    - Similarly, you are going to need [Hoomd-dlext](https://github.com/SSAGESLabs/hoomd-dlext#readme)

- JAX MD is maybe

4. Install pysages

`
pip install git+https://github.com/SSAGESLabs/PySAGES.git
`





## Usage

TODO

## Run experiments

TODO

### Butane (hoomd)

TODO

### Alaline Dipeptide (OpenMM)

TODO

### A protein. IDK yet

TODO

## Implement new models

TODO

## Benchmark models

TODO

## References 



## TODO

- CV Learner

- Jax MD if I have time