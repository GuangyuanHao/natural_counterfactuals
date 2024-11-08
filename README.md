## Natural Counterfactuals With Necessary Backtracking
<!-- [:hugs:Huggingface demo here!:hugs:](https://huggingface.co/spaces/mira-causality/counterfactuals) -->

Code for the **NeurIPS** paper:

>[**Natural Counterfactuals With Necessary Backtracking**](https://arxiv.org/abs/2402.01607)

BibTeX (arXiv for now):
```
@article{hao2024natural,
  title={Natural Counterfactuals With Necessary Backtracking},
  author={Hao, Guang-Yuan and Zhang, Jiji and Huang, Biwei and Wang, Hao and Zhang, Kun},
  journal={NeurIPS},
  year={2024}
}
```
#### This code is based on [casual-gen](https://github.com/biomedia-mira/causal-gen).

The project consists of two directories. The first directory, `src-2o`, includes the following components: code for generating toy datasets, learning Structural Causal Models (SCM) mechanisms for all datasets (including toy datasets, MorphMNIST, and 3DIdentBOX), and code for counterfactual inference, covering both non-backtracking counterfactuals and natural counterfactuals.

The primary distinction between `src-2o` and `src-3m` lies in the distance metrics used: `src-2o` applies distance in observed variables, as discussed in the main paper, whereas `src-3m` utilizes mechanism distance, which is detailed in the appendix.

### Project Structure:

```
📦src-2o                               # main source code directory
 ┣ 📂pgm                               # learn all SCM mechanisms except the image's and counterfactual inference for morphmnist and 3DIdentBOX
 ┃ ┣ 📜dscm.py                         # deep structural causal model Pytorch module
 ┃ ┣ 📜flow_pgm.py                     # Flow mechanisms in Pyro
 ┃ ┣ 📜layers.py                       # utility modules/layers
 ┃ ┣ 📜resnet.py                       # resnet model definition
 ┃ ┣ 📜utils_pgm.py                    # graphical model utilities
 ┃ ┣ 📜train_pgm.py                    # SCM mechanisms training code (Pyro)
 ┃ ┣ 📜train_ours_box.py               # counterfactual inference on 3DIdentBOX (non-backtracking counterfactuals: jdcf; natural counterfactuals: our) 
 ┃ ┣ 📜train_ours_mnist.py             # counterfactual inference on morphmnist (non-backtracking counterfactuals: jdcf; natural counterfactuals: our) 
 ┃ ┣ 📜run_local_morph_oth.sh          # launch script to learn normalizing flows on t and i of morphmnist
 ┃ ┣ 📜run_local_morph_oth_aux.sh      # launch script to learn anti-causal predictor for t and i given image of morphmnist
 ┃ ┣ 📜train_ours_mnist_vae.sh         # launch script to do non-backtracking counterfactuals and natural counterfactuals using simple VAE as the backbone
 ┃ ┣ 📜train_ours_mnist_hvae.sh        # launch script to do non-backtracking counterfactuals and natural counterfactuals using HVAE as the backbone
 ┃ ┣ 📜run_box_pgm.sh                  # launch script to learn normalizing flows on variables except image of Weak-3DIdent 
 ┃ ┣ 📜run_box_aux.sh                  # launch script to learn anti-causal predictor for variables except image of Weak-3DIdent 
 ┃ ┣ 📜run_box2_pgm.sh                 # launch script to learn normalizing flows on variables except image of Strong-3DIdent 
 ┃ ┣ 📜run_box2_aux.sh                 # launch script to learn anti-causal predictor for variables except image of Strong-3DIdent
 ┃ ┣ 📜train_ours_mnist_box.sh         # launch script to do non-backtracking counterfactuals and natural counterfactuals on Weak-3DIdent
 ┃ ┗ 📜train_ours_mnist_box22.sh       # launch script to do non-backtracking counterfactuals and natural counterfactuals on Strong-3DIdent
 ┃ ┣ 📂toy_dataset                     # graphical models for all SCM mechanisms of toy dasets 
 ┃ ┣ 📜toy_pgm.py                      # Flow mechanisms in Pyro
 ┃ ┣ 📜 train_toy2.py                  # counterfactual inference on toy-3 of the paper 
 ┃ ┣ 📜 train_toy3.py                  # counterfactual inference on toy-4 of the paper 
 ┃ ┣ 📜 train_toy4.py                  # counterfactual inference on toy-2 of the paper 
 ┃ ┣ 📜 train_toy5.py                  # counterfactual inference on toy-1 of the paper 
 ┃ ┣ 📜run_pgm_toy2.sh                 # launch script to learn normalizing flows on variables of toy-3 of the paper 
 ┃ ┣ 📜run_pgm_toy3.sh                 # launch script to learn normalizing flows on variables of toy-4 of the paper
 ┃ ┣ 📜run_pgm_toy4.sh                 # launch script to learn normalizing flows on variables of toy-2 of the paper
 ┃ ┣ 📜run_pgm_toy5.sh                 # launch script to learn normalizing flows on variables of toy-1 of the paper
 ┃ ┣ 📜train_toy2.sh                   # launch script to do non-backtracking counterfactuals and natural counterfactuals on toy-3 of the paper
 ┃ ┣ 📜train_toy3.sh                   # launch script to do non-backtracking counterfactuals and natural counterfactuals on toy-4 of the paper
 ┃ ┣ 📜train_toy4.sh                   # launch script to do non-backtracking counterfactuals and natural counterfactuals on toy-2 of the paper
 ┃ ┗ 📜train_toy5.sh                   # launch script to do non-backtracking counterfactuals and natural counterfactuals on toy-1 of the paper 
 ┣ 📜datasets.py                       # dataset definitions
 ┣ 📜dmol.py                           # discretized mixture of logistics likelihood
 ┣ 📜hps.py                            # hyperparameters for all datasets
 ┣ 📜main.py                           # main file
 ┣ 📜simple_vae.py                     # single stochastic layer VAE
 ┣ 📜trainer.py                        # training code for image x's causal mechanism
 ┣ 📜train_setup.py                    # training helpers
 ┣ 📜utils.py                          # utilities for training/plotting
 ┣ 📜vae.py                            # HVAE definition; exogenous prior and latent mediator models 
 ┣ 📜run_local_vae.sh                  # launch script to learn image given variable t and i using VAE for morphmnist
 ┣ 📜run_local_hvae.sh                 # launch script to learn image given variable t and i using HVAE for morphmnist
 ┣ 📜run_local_box_h.sh                # launch script to learn image given other variables using HVAE for Weak-3DIdent
 ┣ 📜run_local_box_h222.sh             # launch script to learn image given other variables using HVAE for Strong-3DIdent
 ┣ 📜toy_data2.ipynb                   # generate dataset toy-3 of the paper
 ┣ 📜toy_data2.ipynb                   # generate dataset toy-4 of the paper
 ┣ 📜toy_data2.ipynb                   # generate dataset toy-2 of the paper
 ┗ 📜toy_data2.ipynb                   # generate dataset toy-1 of the paper
 ```




