#Requirements: fenicsx, pip install: gmsh, pandas, numpy, sympy

#1D-examples
#
#Reproduce experiments with a single contrast
python3 SingleContrast.py 0.5
python3 SingleContrast.py 0.1
#Reproduce experiments for increasing contrast
python3 MultContrast.py fix
python3 MultContrast.py adap

#Generate data for exact solution 
python3 plot_exact.py SJ 2.5 0.25
python3 plot_exact.py SJ 5.5 0.25
python3 plot_exact.py MJ 2.5 0.5
python3 plot_exact.py MJ 7.5 0.25
python3 plot_exact.py MJ 7.5 0.5
python3 plot_exact.py MJ 11.5

#For plots with single jump
python3 ApproxPlots.py

#For plots with multiple jumps
python3 ApproxPlotsMultipleJumps.py 3 7.5
python3 ApproxPlotsMultipleJumps.py 4 7.5
python3 ApproxPlotsMultipleJumps.py 3 11.5
python3 ApproxPlotsMultipleJumps.py 4 11.5
python3 ApproxPlotsMultipleJumpsMultT.py 0.5 2.5
python3 ApproxPlotsMultipleJumpsMultT.py 1.0 2.5
python3 ApproxPlotsMultipleJumpsMultT.py 0.5 7.5
python3 ApproxPlotsMultipleJumpsMultT.py 1.0 7.5

#2D-examples
python3 solve-2D-jump.py
