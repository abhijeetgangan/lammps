import pickle
from mytest.lj import MLIAPLJ

# Argon LJ parameters (metal units): epsilon = 0.010323 eV, sigma = 3.405 Angstrom
argon = MLIAPLJ(["Ar"], epsilon=0.010323, sigma=3.405)

with open("argon_lj.pkl", "wb") as f:
    pickle.dump(argon, f) 