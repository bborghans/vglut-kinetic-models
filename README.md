# Simulation of VGLUT1 channel function or active transport

We provide kinetic models, including ensembles of parameters optimised against electrophysiological recordings, to simulate various functions of WT and H120A VGLUT1 membrane proteins as described in our publication. Time courses of VGLUT1 Cl<sup>-</sup> channel open probabilities are simulated for the WT and H120A mutant protein, via a channel function. Secondary-active glutamate or aspartate transport current is simulated as the charge flux between states based on transport cycles.

## Repository Structure

- **parms_simulation/** 
    - Model simulation using a transition rate matrix. See the [README](./parms_simulation/README.md) for details.
- **parms_optimization/**
    - Parameters optimization. See the [README](./parms_optimization/README.md) for details.

## Installation

We provide environment files for win-64, linux-64, and osx-arm64 platforms defining a conda environment with all the libraries required by the script.

* Download and install, following the instructions,  [conda](https://docs.anaconda.com/miniconda/) (we recommend miniconda).
* Create the environment through the following instruction (replace <your platform> with win-64, linux-64, or osx-arm64): `conda env create -f environment-<your platform>.yml --name vglut`
* Activate the environment `conda activate vglut`

# citation

If you make use of this data, please cite the corresponding paper.  
https://www.biorxiv.org/content/10.1101/2024.09.06.609381 (preprint version)

```
@Article{,
  author          = {},
  journal         = {},
  title           = {},
  year            = {},
  pages           = {},
  volume          = {},
  issue           = {},
  doi             = {}
  url             = {}  
}
```
