# INI ISINYA OPERASI MATRIKS BIAR ENAK
import numpy as np
from scipy.sparse.construct import bmat
from sympy import *
import copy
import cv2
from scipy.linalg import svd

def multiply_matrix(m1,m2):
    col = len(m2[0])
    row = len(m1)
    m3 = [[0 for j in range(col)] for i in range(row)]

    for i in range(0,row):
        for j in range(0,col):
            m3[i][j] = 0
            for k in range(0,len(m1[0])):
                m3[i][j] += m1[i][k] * m2[k][j]

    return m3

def transpose(m):
    col = len(m)
    row = len(m[0])
    hasil = [[0 for j in range(col)] for i in range(row)]

    for i in range(0,row):
        for j in range(0,col):
            hasil[i][j] = m[j][i]
    return hasil

def determinant(m):
    col = len(m[0])
    row = len(m)
    det = 0
    if (row == 2) and (col == 2):
        det = (m[0][0]*m[1][1]) - (m[0][1]*m[1][0])
    else:
        sign = 1
        for i in range(0,col):
            kofaktor = m[0][i]
            valid_cols = [0 for i in range(col-1)]
            valid_col_idx = 0
            for j in range(0,col):
                if (j != i):
                    valid_cols[valid_col_idx] = j
                    valid_col_idx += 1
            m2 = [[0 for j in range(col-1)] for i in range(row-1)]
            for new_row in range(len(m2)):
                for new_col in range(len(m2[0])):
                    valid_col = valid_cols[new_col]
                    m2[new_row][new_col] = m[new_row+1][valid_col]

            det += sign * kofaktor * determinant(m2)
            sign = sign * (-1)

    return det

def substractMat(m1,m2):    #m1-m2
    row = len(m1)
    col = len(m1[0])
    res = [[0 for j in range(col)]for i in range(row)]
    for i in range(row):
        for j in range(col):
            res[i][j] = m1[i][j] - m2[i][j]

    return res

def findEigen(m):
    mat = copy.deepcopy(m)
    while(True):
        q, r = np.linalg.qr(m)
        m = r @ q
        if len(r) >= len(mat[0]):
            break
    # Algortima QR
    row = len(r)
    eig = []
    for i in range(row):
        # print(i)
        x = r[i][i]
        eig.append(abs(round(x)))

    # eig = list(set(eig)) # drop duplicates
    eig.sort(reverse=True)
    return eig


def findEigenVector(eig, m):
    # eig : list of eigen
    # m : matrix input
    arrMat = []
    for ld in eig:
        mat = copy.deepcopy(m)
        #lambdaI-A
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if (i==j):
                    mat[i][j] = (ld-(mat[i][j])) #lambda - elem
                else:
                    mat[i][j] = -mat[i][j]
        for i in range(len(mat)):
            mat[i].append(0)
        arrMat.append(mat)

    resVec = [] #result vector
    for m in arrMat:
        m = rrEchelonForm(m)
        base = findBase(m)
        resVec.append(base)
    return resVec


def displayMatrix(m):
    for i in range(len(m)):
        for j in range(len(m[0])):
            print(m[i][j], end=" ")
        print()

def displaySol(arr):
    for a in arr:
        print(a, end=" ")

def rrEchelonForm(m):
    mat = Matrix(m)
    mat = mat.rref()
    mr = [[0 for j in range(len(m[0]))] for i in range(len(m))]
    a = 0
    b = 0
    for i in range(len(mat[0])):
        mr[a][b] = mat[0][i]
        if (b<len(m[0])-1):
            b+=1
        else:
            a+=1
            b=0
    return mr

def isRowZero(m,i):
    for j in range(len(m[0])):
        if (m[i][j]!=0):
            return False
    return True

def findBase(m):
    base = [[0 for j in range(len(m[0])-1)] for i in range (len(m[0])-1)]
    for i in range(len(m)):
        found = false
        for j in range(len(m[0])-1):
            if m[i][j] == 1 and not found:
                found = true
            elif found and m[i][j]!=0 :
                base[j][i] = -m[i][j]
    for i in range(len(base)):
        if (not(isRowZero(base,i))):
            base[i][i] = 1
    
    base_return = []
    for i in range(len(m)):
        if (not(isRowZero(base,i))):
            base_return.append(base[i])

    return base_return

def findMatrixSigma(m):
    mT = transpose(m)
    M = multiply_matrix(m,mT)
    eig = findEigen(M)
    result = [[0 for j in range(len(m[0]))] for i in range(len(m))]
    for k in range(len(m[0])):
        result[k][k] = eig[k]**0.5
        # print(i)
    return result

def normalize_vector(v):
    v2 = v
    sum_squared = 0
    for elmt in v2:
        sum_squared += elmt**2
    v_length = sqrt(sum_squared)


    return (np.array(v2) / v_length)

def findMatrixVT(m):
    ATA = multiply_matrix(transpose(m),m)
    eigen_values = findEigen(ATA)
    res = []

    arrVec = findEigenVector(eigen_values,ATA)
    for i in range(len(arrVec)):
        normalized_v = normalize_vector(arrVec[i][0])
        res.append(normalized_v)

    return res

def findMatrixU(m):
    Trans = transpose(m)
    mmT = multiply_matrix(m,Trans)
    egen = findEigen(mmT)
    egenvec = findEigenVector(egen,mmT)

    result = []
    for i in range(len(egenvec)):
        hasilnorm_u = normalize_vector(egenvec[i][0])
        result.append(hasilnorm_u)
    return transpose(result)

def ImgToMat(imgpath):
    mat = cv2.imread(imgpath)
    return mat

def compressImg(imgpath, percent):
    m = ImgToMat(imgpath)
    BMat, GMat, RMat = cv2.split(m)
    B = copy.deepcopy(BMat)
    G = copy.deepcopy(GMat)
    R = copy.deepcopy(RMat)
    BMat = BMat.astype(float)
    GMat = GMat.astype(float)
    RMat = RMat.astype(float)
    percent = 100 - percent
    
    #BMat SVD Decomposition
    
    U, S, Vt = svd(BMat)
    k = int((percent/100) * len(U[0]))
    print(k)
    UNew = [[0 for j in range(k)] for i in range(len(U))]
    for i in range(k):
        for j in range(len(U)):
            UNew[j][i] = U[j][i]
    U = UNew

    SNew = [[0 for j in range(k)] for i in range(k)]
    for i in range(k):
        SNew[i][i] = S[i]
    S = SNew

    VtNew = []
    for i in range(k):
        VtNew.append(Vt[i])
    Vt = VtNew
    
    # print(f"{len(U)} x {len(U[0])}")
    # print(f"{len(S)} x {len(S[0])}")
    US = np.matmul(U,S)
    # print(len(US[0]))
    # print(len(Vt))
    BRes = np.matmul(US,Vt)

    #GMat SVD Decomposition
    U, S, Vt = svd(GMat)
    k = int((percent/100) * len(U[0]))
    print(k)
    UNew = [[0 for j in range(k)] for i in range(len(U))]
    for i in range(k):
        for j in range(len(U)):
            UNew[j][i] = U[j][i]
    U = UNew

    SNew = [[0 for j in range(k)] for i in range(k)]
    for i in range(k):
        SNew[i][i] = S[i]
    S = SNew

    VtNew = []
    for i in range(k):
        VtNew.append(Vt[i])
    Vt = VtNew
    
    US = np.matmul(U,S)
    GRes = np.matmul(US,Vt)

    #RMat SVD Decomposition
    U, S, Vt = svd(RMat)
    k = int((percent/100) * len(U[0]))
    print(k)
    UNew = [[0 for j in range(k)] for i in range(len(U))]
    for i in range(k):
        for j in range(len(U)):
            UNew[j][i] = U[j][i]
    U = UNew

    SNew = [[0 for j in range(k)] for i in range(k)]
    for i in range(k):
        SNew[i][i] = S[i]
    S = SNew

    VtNew = []
    for i in range(k):
        VtNew.append(Vt[i])
    Vt = VtNew
    
    US = np.matmul(U,S)
    RRes = np.matmul(US,Vt)
    
    #conversion from float to uint8
    BRes = BRes.astype(np.uint8)
    GRes = GRes.astype(np.uint8)
    RRes = RRes.astype(np.uint8)
    
    #merge BGR
    img4 = cv2.merge([BRes,GRes,RRes])
    # cv2.imshow('window',img4)
    # cv2.waitKey(0)
    cv2.imwrite("comp_amogus.jpg",img4)