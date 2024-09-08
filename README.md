# CURATE: Scaling-up Differentially Private Causal Graph Discovery
This repository contains the code for adaptive differentially private causal graph discovery algorithm CURATE along with the datasets used for the experiments presented in the main paper. 
## Overview
CURATE enables adaptive privacy budget allocation in the setting of differentially private causal graph discovery. 

## Code and Data Description
The used code for constraint-based CURATE algorithm is available in "CURATE_cb.py" file. The score-based CURATE code is available at "CURATE_sb" folder and the datasets used in the experiments are also a part of the main repository. The score based CURATE algorithm is a modified and extended version of [NOLEAKS](https://github.com/pckennethma/NoLeaks-artifact) algorithm.
## Requisites
Python 3.6.10
## Dependency
numpy
pandas
scipy
python-igraph

## Usage
For the implementation of constraint-based CURATE, CURATE_cb.py needs to be run. For score-based CURATE algorithm the usage command is: python -m notears.bnlearn
## References:
The codes for non-private PC and Priv-PC, SVT-PC, EM-PC, and NOLEAKS are borrowed from the following repositories:
[PC Algorithm](https://github.com/keiichishima/pcalg) 
[Priv-PC, SVT-PC, EM-PC](https://github.com/sunblaze-ucb/Priv-PC-Differentially-Private-Causal-Graph-Discovery)
[NOLEAKS](https://github.com/pckennethma/NoLeaks-artifact)
