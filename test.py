# INI BUAT COBA-COBA AJA
import numpy as np
from tabulate import tabulate

from OperasiMatriks import *
from sympy import *
from scipy.linalg import svd

#find eigen vector for one matrix
# m = [
#     [11,1],
#     [1,11]
# ]
# eig = findEigen(m)
# print(eig)
# arrVec = findEigenVector(eig, m)
# print(arrVec)
# print(normalize_vector([1,1]))

#mat = [[11,1],[1,11]]
# eig = findEigen(mat)
# print(eig)

#sigma mat tester
# A = [[3,1,1],[-1,3,1]]
# sigma = findMatrixSigma(A)
# VT = findMatrixVT(A)
# print(sigma)
# displayMatrix(VT)
# print("")
# X = findMatrixU(A)
# displayMatrix(X)

#opencv tester
path1 = "amogus.jpg"
path2 = "test.jpeg"
path3 = "pixel.png"
# mat = ImgToMat(path1)
#column - row - element (B-G-R) each elem
# print(mat[499])
# print(mat[497][490]) #row - colum
# print(mat[0][0])
# print(mat[0][0][0])
# print(len(mat))
# blueMat = getBlueMat(mat)
# greenMat = getGreenMat(mat)
# redMat = getRedMat(mat)
# print(blueMat[497][490]) #row - column
# print(greenMat[497][490])
# print(redMat[497][490])
# b,g,r = cv2.split(mat)
# print(g[497][490])
# img = cv2.imread(path1)
# img[0:150, 0:100] = [0,255,0] #row 0 - 150, 
# cv2.imshow('window',img)
# cv2.waitKey(0)

#compress 95%
compressImg(path1, 99)


# b,g,r = cv2.split(mat)
# img = cv2.merge([b,g,r])
# cv2.imshow('w',img)
# cv2.waitKey(0)

# U, s, VT = svd(b)
# print(U)
# print(s)
# print(VT)


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