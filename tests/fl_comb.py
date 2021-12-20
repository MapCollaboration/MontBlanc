import numpy as np
from numpy.linalg import inv
from sympy import *

#---- SIAonly case

# number of output NN nodes
Nout = 5

# flavour map as implemented in our runcard
fl_list = [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0, # - g
            0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0, # - d+ + s+
            0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0, # - u+
            0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, # - c+
            0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0] # - b+

# from fl_list to array
rows,cols = Nout, int(len(fl_list)/Nout)
fl_array = np.resize(fl_list,(rows,cols))

# Compute the (Moore-Penrose) pseudo-inverse of a matrix.
fl_array_inv = np.linalg.pinv(fl_array)
fl_matrix_inv = np.matrix(fl_array_inv.round(3))

# fl as symbols
tb,  bb,  cb,  sb,  ub,  db,  g,   d,   u,  s,   c,   b,   t  = symbols('tb  bb  cb  sb  ub  db  g   d   u   s   c   b   t')

# NN parameterisation as combination of flavours
NN_basis = np.transpose(np.matrix([g, sb+db+d+s,ub+u,cb+c,bb+b]))

print("flavour map\n C = ", fl_array)
print("\n")
print("flavour map (Moore-Penrose) pseudo-inverse\n Cp = ", fl_array_inv.round(3))



phys_basis = fl_matrix_inv*NN_basis
print("\n")
print("transition physical basis\n Cp*NN = Cp*[g, sb+db+d+s, ub+u,cb+c, bb+b] = \n",phys_basis)

print("\n To input in the config card:")
flavourmap = np.transpose(fl_matrix_inv).flatten().tolist()[0]
print(flavourmap)
print("dimension = ", int(len(flavourmap)/13),13)

    
