# INI BUAT COBA-COBA AJA
import numpy as np
from tabulate import tabulate

from OperasiMatriks import *
from sympy import *

#find eigen vector for one matrix
m = [
    [10,0,2],
    [0,10,4],
    [2,4,2]
]
eig = findEigen(m)
print(eig)
arrVec = findEigenVector(eig, m)
print(arrVec)

# mat = [[11,1],[1,11]]
# eig = findEigen(mat)
# print(eig)


# x = InverseSPL(mat,[[0],[0],[0]])
# displayMatrix(matInverse(mat))
# displayMatrix(x)

# m = [[3,1,1],[-1,3,1]]

# m2 = np.array([[3,-2,0],[-2,3,0],[0,0,5]])
# q,r = np.linalg.qr(m2)

# for i in range(1000):
#     q, r = np.linalg.qr(m2)
#     m2 = r @ q

# print("Q")
# print(tabulate(q))
# print("R")
# print(tabulate(r))
# print(tabulate(r@q))
# print(findEigen(m2))