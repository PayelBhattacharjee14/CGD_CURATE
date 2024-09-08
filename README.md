# CURATE: Scaling-up Differentially Private Causal Graph Discovery
This repository contains the implementation code of adaptive privacy budgeting causal graph discovery algorithm CURATE along with the supplementary document that provides the proof of the paper.
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
## References:
The codes for non-private PC and Priv-PC, SVT-PC, EM-PC, and NOLEAKS are borrowed from the following repositories:
[PC Algorithm](https://github.com/keiichishima/pcalg) 
[Priv-PC, SVT-PC, EM-PC](https://github.com/sunblaze-ucb/Priv-PC-Differentially-Private-Causal-Graph-Discovery)
[NOLEAKS](https://github.com/pckennethma/NoLeaks-artifact)
