# Sonic on Ray

This file describes how to use Sonic with Ray and RLLib. We include
instructions on how to get the training running on EC2.

## Running training on a single node

Start a p2.8xlarge with the Deep Learning AMI (Ubuntu).

Activate the TensorFlow environment with

```
source activate tensorflow_p36
```

Now install Ray and the RLlib requirements using

```
pip install ray opencv-python
```

Next we need to install the gym retro environment. Run

```
git clone --recursive git@github.com:openai/retro.git gym-retro
cd gym-retro
pip install -e .
```

Now clone this repo:

```
cd ~
git clone git@github.com:openai/sonic-on-ray.git
```

You can then run the training with

```
cd sonic-on-ray
python train_ppo.py
```
