# INI ISINYA OPERASI MATRIKS BIAR ENAK
import numpy as np
from sympy import *
import copy

from sympy.polys.polyoptions import Sort

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

#modified cofactor determinant with element as equation
def modifiedDetCof(m):
    row = len(m)
    col = len(m[0])
    if (row==1 and col==1):   #base 1
        return m
    elif (row==2 and col==2): #base 2
        max_degree1 = (len(m[0][0])-1) + (len(m[1][1])-1)
        max_degree2 = (len(m[0][1])-1) + (len(m[1][0])-1)
        if (max_degree1 > max_degree2):
            arr = [0 for i in range(max_degree1+1)]
        else:
            arr = [0 for i in range(max_degree2+1)]
        #[0][0] * [1][1]
        for i in range(len(m[0][0])):
            for j in range(len(m[1][1])):
                arr[i+j] += m[0][0][i]*m[1][1][j]
        
        #[0][1] * [1][0]
        for i in range (len(m[0][1])):
            for j in range(len(m[1][0])):
                arr[i+j] = arr[i+j] - (m[0][1][i]*m[1][0][j])
        return arr
    else:   #recursive
        print()
        #in progress

#solve equation, input array =  [x^0, x^1, x^2,...]
def solveEquation(arr):
    if len(arr)==1:
        return arr[0]
    elif len(arr)==2: #[x^0,x^1] 
        return arr[1]
    elif len(arr)==3: #[x^0,x^1,x^2]
        D = (arr[1]**2)-(4*arr[2]*arr[0]) #discriminant
        if (D<0):
            return []
        else:
            solution = []
            sqrtD = D**0.5
            s1 = (-arr[1]+sqrtD)/(2*arr[2])
            solution.append(s1)
            s2  = (-arr[1]-sqrtD)/(2*arr[2])
            solution.append(s2)
            return solution
    else:
        print()
        #in progressm, higher nth degree polynomial


def eigenFinderMxN(m):
    A = m
    At = transpose(m)
    AAt = multiply_matrix(A,At) #MxM
    row = len(AAt)
    col = row

    #[lambda^0,lambda^1,lambda^2,...]
    LambdaIminusA = [[[0,0] for j in range(col)]for i in range(row)]
    for i in range(row):
        for j in range(col):
            if (i==j):
                LambdaIminusA[i][j][1] = 1
                LambdaIminusA[i][j][0] = -AAt[i][j]
            else:
                LambdaIminusA[i][j][1] = 0
                LambdaIminusA[i][j][0] = -AAt[i][j]
    det = modifiedDetCof(LambdaIminusA)
    sol = solveEquation(det)
    return sorted(sol, reverse=True)

def findEigen(m):
    for i in range(5000):
        q, r = np.linalg.qr(m)
        m = r @ q

    # Algortima QR
    row = len(r)
    eig = []
    for i in range(row):
        x = r[i][i]
        eig.append(abs(round(x)))

    eig = list(set(eig)) # drop duplicates
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

def InverseSPL(m,b):
    MInverse = matInverse(m)
    x = multiply_matrix(MInverse, b)
    return x

def matCofactor(m):
    mc = [[0 for j in range(len(m[0]))] for i in range(len(m))]
    a = 0
    b = 0
    multiply = 1
    for i in range(len(m)):
        if (i%2==0):
            multiply = 1
        else:
            multiply = -1
        for j in range(len(m[0])):
            m1 = [[0 for j in range(len(m[0])-1)] for i in range(len(m)-1)]
            for c in range(len(m)):
                for d in range(len(m[0])):
                    if (c!=i and d!=j):
                        m1[a][b] = m[c][d]
                        if (b+1 < len(m1[0])):
                            b+=1
                        elif (a+1 < len(m1)):
                            a+=1
                            b = 0
            a = 0
            b = 0
            deter = determinant(m1)
            mc[i][j] = multiply*deter
            multiply*=-1
    return mc
    

def adj(m):
    if (len(m)==1 and len(m[0]) ==1):
        m[0][0] = 1
        return m
    else:
        madj = transpose(matCofactor(m))
        return madj


def matInverse(m):
    deter = determinant(m)
    MInverse = adj(m)
    for i in range(len(MInverse)):
        for j in range(len(MInverse[0])):
            MInverse[i][j]*=(1/deter)
    return MInverse

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
    pop_rowIndex = []
    for i in range(len(base)):
        if (not(isRowZero(base,i))):
            base[i][i] = 1
        else:
            pop_rowIndex.append(i)
    
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
    for i in range(len(m)):
        result[i][i] = eig[i]**0.5
    return result