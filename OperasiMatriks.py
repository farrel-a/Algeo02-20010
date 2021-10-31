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

def eigenFinderMxN(m):
    A = m
    At = transpose(m)
    AAt = multiply_matrix(A,At) #MxM
    row = len(AAt)
    col = row

    LambdaIminusA = [[[0,0] for j in range(col)]for i in range(row)] #[[lambda^0,lambda^1,lambda^2,...], konstanta]
    for i in range(row):
        for j in range(col):
            if (i==j):
                LambdaIminusA[i][j][1] = 1
                LambdaIminusA[i][j][0] = -AAt[i][j]
            else:
                LambdaIminusA[i][j][1] = 0
                LambdaIminusA[i][j][0] = -AAt[i][j]
    print(LambdaIminusA)
    #in progress