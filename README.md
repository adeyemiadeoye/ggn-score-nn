# Regularized Gauss-Newton for learning overparameterized neural networks
 The `ggn-score-nn` repository contains code that accompany the paper: "Regularized Gauss-Newton for Learning Overparameterized Neural Networks." The code implements a training procedure for two-layer neural networks using the `GGN-SCORE` algorithm from [`SelfConcordantSmoothOptimization.jl`](https://github.com/adeyemiadeoye/SelfConcordantSmoothOptimization.jl). The `GD` algorithm used for comparison is also based on the implementation from `SelfConcordantSmoothOptimization.jl`. These are achieved by setting the `use_prox=false` option in each of `ggn_optimizer = ProxGGNSCORE(use_prox=false)` and `gd_optimizer = ProxGrad(use_prox=false)`, respectively.

 ## Abstract
 The generalized Gauss-Newton (GGN) optimization method incorporates curvature estimates into its solution steps, and provides a good approximation to the Newton method for large-scale optimization problems. GGN has been found particularly interesting for practical training of deep neural networks, not only for its impressive convergence speed, but also for its close relation with neural tangent kernel regression, which is central to recent studies that aim to understand the optimization and generalization properties of neural networks. This work studies a GGN method for optimizing a two-layer neural network with explicit regularization. In particular, we consider a class of generalized self-concordant (GSC) functions that provide smooth approximations to commonly-used penalty terms in the objective function of the optimization problem. This approach provides an adaptive learning rate selection technique that requires little to no tuning for optimal performance. We study the convergence of the two-layer neural network, considered to be overparameterized, in the optimization loop of the resulting GGN method for a given scaling of the network parameters. Our numerical experiments highlight specific aspects of GSC regularization that help to improve generalization of the optimized neural network.

## Contents
- `src/NNOptimization.jl`: implements the main training loop for the synthetic data, based on `SelfConcordantSmoothOptimization.jl`.
- `experiements/`: contains main files to run for the experimental results in the paper.
  - `utils/`: contains utility files for the experiments.
  - `data/`: contains the manually downloaded data for the experiments.
  - `tmp/`: results are stored here if the option is set.
  - `figures/`: plots will be stored here.

Each experiment file can be run directly. Specific options in an experiment file can be set in the last lines of the code.

## Dependencies
A complete list of Julia package dependencies can be found in `Project.toml`.

To successfully run the experiments code, it is required to install the latest version of `SelfConcordantSmoothOptimization.jl`, currently at `v0.1.4`.