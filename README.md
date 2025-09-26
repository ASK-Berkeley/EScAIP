# EScAIP: Efficiently Scaled Attention Interatomic Potential

## Note: EScAIP is now integrated into the [FairChem](https://github.com/FAIR-Chem/fairchem) repository. This repository contains checkpoints and usage examples, but is no longer maintained.

This repository contains the official implementation of the [Efficiently Scaled Attention Interatomic Potential (NeurIPS 2024)](https://openreview.net/forum?id=Y4mBaZu4vy).

> Scaling has been a critical factor in improving model performance and generalization across various fields of machine learning.
It involves how a model’s performance changes with increases in model size or input data, as well as how efficiently computational resources are utilized to support this growth.
Despite successes in scaling other types of machine learning models, the study of scaling in Neural Network Interatomic Potentials (NNIPs) remains limited. NNIPs act as surrogate models for ab initio quantum mechanical calculations, predicting the energy and forces between atoms in molecules and materials based on atomic configurations. The dominant paradigm in this field is to incorporate numerous physical domain constraints into the model, such as symmetry constraints like rotational equivariance. We contend that these increasingly complex domain constraints inhibit the scaling ability of NNIPs, and such strategies are likely to cause model performance to plateau in the long run. In this work, we take an alternative approach and start by systematically studying NNIP scaling properties and strategies. Our findings indicate that scaling the model through attention mechanisms is both efficient and improves model expressivity. These insights motivate us to develop an NNIP architecture designed for scalability: the Efficiently Scaled Attention Interatomic Potential (EScAIP).
EScAIP leverages a novel multi-head self-attention formulation within graph neural networks, applying attention at the neighbor-level representations.
Implemented with highly-optimized attention GPU kernels, EScAIP achieves substantial gains in efficiency---at least 10x speed up in inference time, 5x less in memory usage---compared to existing NNIP models. EScAIP also achieves state-of-the-art performance on a wide range of datasets including catalysts (OC20 and OC22), molecules (SPICE), and materials (MPTrj).
We emphasize that our approach should be thought of as a philosophy rather than a specific model, representing a proof-of-concept towards developing general-purpose NNIPs that achieve better expressivity through scaling, and continue to scale efficiently with increased computational resources and training data.

## Install

Install FairChem (see [FairChem documentation](https://fair-chem.github.io/core/install.html)). The model is now contained in the package.

## Train

Now training is done through FairChem CLI. For example, to train a model on the OC20 dataset:

1. Change the user-dependent configs in the FairChem repo, including:
- `configs/escaip/training/cluster/your_cluster.yml`: cluster configuration, including rundir.
- `configs/escaip/training/dataset/your_oc20_dataset.yml`: data path on your machine.
- `configs/escaip/training/oc20_direct_escaip_fair.yml`: the wandb project name and entity.

2. Run the training command:
```bash
fairchem -c configs/escaip/training/oc20_direct_escaip_fair.yml
```

## Simulation

Use FairChem Calculator to simulate with the models. See the [FairChem documentation](https://fair-chem.github.io/core/quickstart.html) for more details.

## Model Architecture

Refer to the [model architecture](model_architecture.md) for more details.

Note: we have seen some performance difference when using different versions of attention kernels. This issue is fixed now. Please refer to the Appendix C of the [updated paper](https://arxiv.org/abs/2410.24169) for more details.

## Pretrained Models

We will release the pretrained models in the paper here.

## Citation

If you find this work useful, please consider citing the following:

```
@inproceedings{
qu2024the,
title={The Importance of Being Scalable: Improving the Speed and Accuracy of Neural Network Interatomic Potentials Across Chemical Domains},
author={Eric Qu and Aditi S. Krishnapriyan},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=Y4mBaZu4vy}
}
```
