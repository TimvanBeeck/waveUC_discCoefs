


This repository contains instructions for reproducing the numerical experiments in the paper 
> "Variational data assimilation for the wave equation in heterogeneous media: Numerical investigation of stability"
>
> * Authors: Erik Burman(1), Janosch Preuss(2), Tim van Beeck(3)
> * (1): University College London
> * (2): Inria Bordeaux (project team: Makutu)
> * (3): University of Göttingen

## Requirements 
The numerical examples use the finite element software [fenicsx](https://fenicsproject.org/) (version 0.7.0). The file `environment.yml` allows to create a conda environment in which the code can be executed via `conda env create -f environment.yml`. 

## Reproduction
To reproduce all the experiments, go into the folder and execute the bash file `reproduce.sh`, which executes all scripts one after another. The data is written into the `data` directory. 