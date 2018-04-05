# Sonic on Ray

This file describes how to use Sonic with Ray and RLLib. We include
instructions on how to get the training running on EC2.

## Running training on a single node

Start a p2.8xlarge with the Deep Learning AMI (Ubuntu). In us-west-2, this is
ami-d2c759aa.

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

Now clone this repo and install it:

```
cd ~
git clone git@github.com:openai/sonic-on-ray.git
cd sonic-on-ray
pip install -e .
```

You can then run the training with

```
cd ~/sonic-on-ray
python train_ppo.py
```

## Running training on a cluster

First install Ray on your laptop with

```
pip install ray
```

Now clone the sonic-on-ray repo with

```
git clone git@github.com:openai/sonic-on-ray.git
```

And start a cluster with

```
ray create_or_update sonic-autoscaler.yaml
```

After the cluster has been started, you will see a message like this:

```
Started Ray on this node. You can add additional nodes to the cluster by calling

    ray start --redis-address 172.31.58.176:6379

from the node you wish to add. You can connect a driver to the cluster from Python by running

    import ray
    ray.init(redis_address="172.31.58.176:6379")

[...]

To login to the cluster, run:

      ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem ubuntu@54.152.27.84
```

You can now start the hyperparameter search by sshing into the cluster, running

```
source activate tensorflow_p36
```

and replacing the `ray.init()` call in `~/sonic-on-ray/train_ppo_grid_search.py`
by the one printed above and then running the script.
