# INI ISINYA OPERASI MATRIKS BIAR ENAK
import numpy as np
from sympy import *
import copy
import cv2
from numpy.linalg import norm
from random import normalvariate
from math import sqrt


def transpose(m):
    # return transpose of m
    col = len(m)
    row = len(m[0])
    hasil = [[0 for j in range(col)] for i in range(row)]

    for i in range(0,row):
        for j in range(0,col):
            hasil[i][j] = m[j][i]
    return hasil

def findEigen(m):
    # return eigen values
    for i in range(50):
        q, r = np.linalg.qr(m)
        m = r @ q
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
            np.append(mat[i],0)
        arrMat.append(mat)

    resVec = [] #result vector
    for m in arrMat:
        m = rrEchelonForm(m)
        base = findBase(m)
        resVec.append(base)
    return resVec


def displayMatrix(m):
    # display matrix elements
    for i in range(len(m)):
        for j in range(len(m[0])):
            print(m[i][j], end=" ")
        print()

def displaySol(arr):
    # display solution
    for a in arr:
        print(a, end=" ")

def rrEchelonForm(m):
    # retrurn row reduced echelon form of m
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
    # return whether row-i in m is zero row
    for j in range(len(m[0])):
        if (m[i][j]!=0):
            return False
    return True

def findBase(m):
    # return base of M
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
    # return sigma matrix of SVD
    mT = transpose(m)
    M = np.matmul(m,mT)
    eig = findEigen(M)
    if (len(m[0])<len(m)):
        result = [[0.0 for j in range(len(m[0]))] for i in range(len(m))]
        for k in range(len(m[0])):
            result[k][k] = eig[k]**0.5
            # print(i)
    else:
        result = [[0.0 for j in range(len(m))] for i in range(len(m[0]))]
        for k in range(len(m)):
            result[k][k] = eig[k]**0.5
            # print(i)

    return result

def normalize_vector(v):
    # return normalized vector
    v2 = v
    sum_squared = 0
    for elmt in v2:
        sum_squared += elmt**2
    v_length = sqrt(sum_squared)


    return (np.array(v2) / v_length)


def ImgToMat(imgpath):
    # return matrix of image pixel
    mat = cv2.imread(imgpath)
    return mat

def compressImg(imgpath, percent_compression):
    # Generate matrix from imgpath
    m = ImgToMat(imgpath)

    # Splig B, G, and R matrix
    BMat, GMat, RMat = cv2.split(m)

    # Conversion to float
    BMat = BMat.astype(float)
    GMat = GMat.astype(float)
    RMat = RMat.astype(float)

    # 0% compression is equal to not compressed picture
    if (percent_compression == 0):
        k = len(BMat[0])
    
    # 100% compression is equal to k = 1 compression
    elif (percent_compression == 100):
        k = 1
    
    # percentage calculation
    else:
        percent_image = 100 - percent_compression
        k = int(((percent_image/100)*len(BMat)*len(BMat[0]))/(len(BMat)+1+len(BMat[0])))
    
    print(f"Computing for {percent_compression}% compression")
    print(f"Computing for K = {k}")

    # BMat SVD Decomposition
    S = findMatrixSigma(BMat)
    
    U, Vt = findMatrixUandVt(BMat,k)

    SNew = [[0 for j in range(k)] for i in range(k)]
    for i in range(k):
        SNew[i][i] = S[i][i]
    S = SNew

    US = np.matmul(U,S)
    BRes = np.matmul(US,Vt)
    print("B matrix computation success")


    # GMat SVD Decomposition
    S = findMatrixSigma(GMat)
    
    U, Vt = findMatrixUandVt(GMat,k)

    SNew = [[0 for j in range(k)] for i in range(k)]
    for i in range(k):
        SNew[i][i] = S[i][i]
    S = SNew

    US = np.matmul(U,S)
    GRes = np.matmul(US,Vt)
    print("G matrix computation success")


    # RMat SVD Decomposition
    S = findMatrixSigma(RMat)
    
    U, Vt = findMatrixUandVt(RMat,k)

    SNew = [[0 for j in range(k)] for i in range(k)]
    for i in range(k):
        SNew[i][i] = S[i][i]
    S = SNew
    
    US = np.matmul(U,S)
    RRes = np.matmul(US,Vt)
    print("R matrix computation Success")
    
    print("BGR matrix computation success !")

    # conversion from float to uint8
    BRes = BRes.astype(np.uint8)
    GRes = GRes.astype(np.uint8)
    RRes = RRes.astype(np.uint8)
    
    # merge BGR
    img = cv2.merge([BRes,GRes,RRes])

    # generate new filename
    filename = filenamecvt(imgpath)

    # save picture
    cv2.imwrite(filename,img)

    # notify
    print(f"File saved as {filename} in test folder !")

def filenamecvt(name):
    #convert filename to add "_compressed" at the end of filename
    imgname = ""
    i = 0
    while(name[i]!="."):
        imgname += name[i]
        i+=1
    ext = ""
    i = -1
    while(name[i]!="."):
        ext += name[i]
        i-=1
    extension=""
    for i in range(len(ext)):
        extension+=ext[len(ext)-1-i]
    filename = f"{imgname}_compressed.{extension}"
    return filename


def randUnitVec(n):
    unnormalized = []
    for i in range(n):
        x = normalvariate(0,1)
        unnormalized.append(x)

    sum = 0
    for x in unnormalized:
        sum += x**2

    norm = sqrt(sum)
    result = []

    for i in range(n):
        res = unnormalized[i]/norm
        result.append(res)

    return result


def svd1d(A, epsilon):
    # SVD computation for 1D
    m = len(A)    # row size
    n = len(A[0]) # col size
    minimum = min(m,n)
    x = randUnitVec(minimum)
    currV = x

    B = np.dot(A.T, A) if (m > n) else np.dot(A, A.T)

    iter = 0
    flag = True
    while flag:
        lastV = currV
        currV = (np.dot(B, lastV)) / (norm(np.dot(B, lastV)))
        if abs(np.dot(currV, lastV)) > 1 - epsilon:
            flag = False
        iter += 1
    return currV


def findMatrixUandVt(A, k):
    # return U matrix and Vt matrix
    epsilon = 1e-10
    A = np.array(A, dtype=float)
    m = len(A)    # row size
    n = len(A[0]) # col size
    svdArr = []

    #power iteration
    i = 0
    flag = True
    while (flag):
        matrix1D = A.copy()

        for u, singularValue,v in svdArr[:i]:
            matrix1D -= singularValue * np.outer(u, v)

        if m <= n:
            u = svd1d(matrix1D, epsilon) 
            sig = norm(np.dot(A.T, u)) 
            v = (np.dot(A.T, u)) / (norm(np.dot(A.T, u)))
        else:
            v = svd1d(matrix1D, epsilon)  
            sig = norm(np.dot(A, v)) 
            u = (np.dot(A, v)) / (norm(np.dot(A, v)))


        svdArr.append((u, sig, v))
        i+=1
        if (i>=k):
            flag = False
            # quit loop

    #result
    u, singval, v = [np.array(x) for x in zip(*svdArr)]
    U = u.T
    Vt = v
    return U, Vt