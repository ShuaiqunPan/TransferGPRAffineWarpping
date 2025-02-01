# TransferGPRAffineWarpping

## Overview
This GitHub repository hosts the implementation for the paper titled "Transfer Learning of Surrogate Models: Integrating Domain Warping and Affine Transformations", submitted to the Genetic and Evolutionary Computation Conference (GECCO 2025). The paper is currently under review.

## Abstract
Surrogate models provide efficient alternatives to computationally demanding real-world processes but often require large datasets for effective training. A promising solution to this limitation is the transfer of pre-trained surrogate models to new tasks. Previous studies have investigated the transfer of differentiable and non-differentiable surrogate models, typically assuming an affine transformation between the source and target functions. This paper extends previous research by addressing a broader range of transformations, including linear and nonlinear variations. Specifically, we consider the combination of an unknown input warping—such as one modeled by the beta cumulative distribution function—with an unspecified affine transformation. Our approach achieves transfer learning by employing a limited number of data points from the target task to optimize these transformations, minimizing empirical loss on the transfer dataset. We validate the proposed method on the widely used Black-Box Optimization Benchmark (BBOB) testbed and a real-world transfer learning task from the automobile industry. The results underscore the significant advantages of the approach, revealing that the transferred surrogate significantly outperforms both the original surrogate and the one built from scratch using the transfer dataset, particularly in data-scarce scenarios.

## Installation
conda env create -f environment.yml

## Usage
python Riemannian_transferGP.py

## Note
For this project, the GPy folder contains unmodified source code from the GPy library (https://gpy.readthedocs.io/en/deploy/), which is utilized for building the Gaussian process regression model.