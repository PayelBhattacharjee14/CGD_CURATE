# CURATE: Scaling-up Differentially Private Causal Graph Discovery
This repository contains the implementation code of adaptive privacy budgeting causal graph discovery algorithm CURATE along with the supplementary document that provides the proof of the paper.
## Overview
CURATE enables adaptive privacy budget allocation in the setting of differentially private causal graph discovery. 

## Supplementary Document
The proof of Lemma-1 presented in the paper along with the sensitivity analysis of the Kendall's Tau test statistic used in the paper is available in the "MLSP_2024_CURATE_supplimentary_document.pdf" file. 

## Code and Data Description
The used code for CURATE algorithm is available in "CURATE.py" file and the datasets used in the experiments are also a part of the main repository.
## Requisites
Python 3.6.10
## Dependency
numpy
pandas
scipy
## References:
The codes for non-private PC and Priv-PC, SVT-PC, EM-PC are borrowed from the following repositories:
[PC Algorithm](https://github.com/keiichishima/pcalg) 
[Priv-PC, SVT-PC, EM-PC](https://github.com/sunblaze-ucb/Priv-PC-Differentially-Private-Causal-Graph-Discovery)
