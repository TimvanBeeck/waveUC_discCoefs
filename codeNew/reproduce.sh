#Requirements: fenicsx, gmsh (!), pandas, numpy....
#Reproduce experiments with a single contrast
python3 SingleContrast.py 0.5
python3 SingleContrast.py 0.1
#Reproduce experiments for increasing contrast
python3 MultContrast.py fix
python3 MultContrast.py adap
