Riemannian Graph Neural Networks with Mixture of Experts (RGNN-MoE)
Official code repository for training and evaluating the Riemannian Graph Neural Network with Mixture of Experts (RGNN-MoE).

Introduction
This repository contains the implementation of the RGNN-MoE model. By leveraging Riemannian manifolds and a Mixture of Experts (MoE) routing mechanism (Gate3), our model efficiently accommodates the structural diversity of molecular graphs for robust Out-Of-Distribution (OOD) generalization.

Datasets
We evaluate our model on two primary benchmarks:

OGB Benchmark: Molecular property prediction (e.g., ogbg-molbace).

DrugOOD Benchmark: OOD generalization on drug discovery datasets (e.g., ic50_assay).

Requirements
The code has been tested under the following environment.

Core Dependencies:

python == 3.9.23

cuda == 12.1

torch == 2.2.0

torch-geometric (pyg) == 2.0.3

geoopt == 0.5.0

numpy == 1.26.4

networkx == 3.0

ogb == 1.3.6

rdkit == 2021.09.4

scikit-learn == 1.6.1

drugood == 0.0.1

To install the exact dependencies, you can recreate the environment using conda/pip or directly install the versions listed above. (Note: To install the drugood package, please refer to the official DrugOOD repository).

How to Run
1. Training on OGB Datasets
To train the RGNN-MoE model on the OGB benchmark (e.g., ogbg-molbace), run the following command:

Bash
python main_ood.py \
  --sub_backend config/rgnn-moe.json \
  --dataset ogbg-molbace \
  --batch_size 32 \
  --epoch 400 \
  --device 0 \
  --patience 400 \
  --lr_riemann 1e-3 \
  --lr 0.005 \
  --out_dim_moe 32 \
  --init_tau 1.5 \
  --sharp_coef 1e-2 \
  --div_coef 1e-2 \
  --proto_std 0.05
2. Training on DrugOOD Datasets
To train the model on the DrugOOD benchmark (e.g., ic50-assay), use the main_drugood.py script:

Bash
python main_drugood.py \
  --data_config config/data_ic50_assay.py \
  --dataset_tag ic50-assay \
  --device 0 \
  --epochs 400 \
  --init_curvs 0.0 -1.0 1.0 \
  --hidden_dim 32 \
  --out_dim 32 \
  --lr_gate 5e-3 \
  --lr_cls 1e-3 \
  --lr_riemann 1e-4 \
  --seed 2022
Folder Specification
config/: Contains JSON and Python configuration files for model backends and DrugOOD dataset settings.

modules/: Contains the core RGNN-MoE model definitions, geometric routing mechanisms (Gate3), and data preprocessing scripts.

main_ood.py: Main script to train and evaluate the model on OGB datasets.

main_drugood.py: Main script to train and evaluate the model on DrugOOD datasets.
