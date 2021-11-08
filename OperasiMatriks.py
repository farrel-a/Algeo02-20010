# INI ISINYA OPERASI MATRIKS BIAR ENAK

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