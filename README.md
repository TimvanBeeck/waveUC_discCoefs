


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

## Reproduction of final experiments
The final experiements of the article uses the software software [ngsolve](https://github.com/NGSolve/ngsolve) (commit: f6f7fcf3ecf5c52a4d6dbdcec95bec925770ed47) and the 
Add-On [ngsxfem](https://github.com/ngsxfem/ngsxfem) (commit: efd8b752fca4045e894532621df232ce209673cd). Please have a look [here](https://ngsolve.org/installation.html) and [here](https://github.com/ngsxfem/ngsxfem/blob/master/INSTALLATION.md) for installation instructions. 

To reproduce the date for the experiments change to the folder `/code/ngsolve` and run 

    python3 python3 whispering-gallery.py 1 1
    python3 python3 whispering-gallery.py 1 0
    python3 python3 whispering-gallery.py 10 1
    python3 python3 whispering-gallery.py 10 0

