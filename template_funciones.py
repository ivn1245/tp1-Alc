import numpy as np
import scipy

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)


#dist=[[0,1,4,2],[1,0,3,8],[4,3,0,8],[2,8,8,0]]
#distancia=np.asarray(dist)
#print(construye_adyacencia(distancia,2))

def calculaLU(matriz):
    # matriz es una matriz de NxN
    lista_de_triangulacion=[]
    U=matriz.copy()
    m=matriz.shape[0]
    n=matriz.shape[1]
    res=[]
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    L = np.eye(n,n)
    for j in range (0, n-1):
        for i in range (j+1,n):
            coef= U[i][j] / U[j][j]
            L[i][j] = coef
            for k in range (n):
                U[i][k]= U[i][k] - (coef * U[j][k])
    res.append(L)
    res.append(U)
    return res

# Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    # Completar! Have fun
def calcular_transpuesta(M):
    m=M.shape[0]
    n=M.shape[1]
    L=np.eye(n,m)
    for i in range (m):
        for j in range (n):
            L[j][i]= M[i][j]
    return L

def calcular_inversa(A):  #calcula la inversa con la implementacion de LU
    dimA= np.eye(A.shape[0])
    L,U=calculaLU(A)
    I=np.identity(dimA)
    primer_sistema= scipy.linalg.solve_triangular(L,I)
    matriz_inversa= scipy.linalg.solve_triangular(U,primer_sistema)
    return matriz_inversa

def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    dimA=A.shape[0]
    for i in range(dimA):
        A[i][i] = 0

    Kinv = np.zeros((dimA, dimA))

    for i in range(dimA):
        suma = 0
        for j in range(dimA):
            suma += A[i][j]
        if suma > 0:
            Kinv[i][i] = 1 / suma
    C = calcular_transpuesta(A) @ Kinv
    return C

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = len(A) # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (1-alfa)*C #primera parte de la ecuacion del pagerank
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = np.ones((1,N)) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    b = (alfa/N)*b
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    F = 1/D
    np.fill_diagonal(F,0)
    dimD=D.shape[0]
    
    Kinv=np.zeros((dimD,dimD)) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    
    for i in range (dimD):
        total_fila=0
        for j in range (dimD):
            total_fila+=F[i][j]
        Kinv[i][i]= 1 / total_fila

    C = F @ Kinv  # Calcula C multiplicando Kinv y F
    return C 

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    C_potencia = np.eye(C.shape[0])
    B = np.eye(C.shape[0])
    if cantidad_de_visitas == 1:
        B += C

    for i in range(1,cantidad_de_visitas):
        # Sumamos las matrices de transición para cada cantidad de pasos    
        C_potencia = C_potencia @ C  # Multiplicamos una vez por C cada vuelta
        B += C_potencia  # Sumamos la nueva potencia
    return B

def calculaV(B,W):
    L,U = calculaLU(B)
    Up = scipy.linalg.solve_triangular(L,W,lower=True) # Primera inversión usando L
    v = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return v

def calcNorma(v):
    norma=0
    for elem in v:
        norma+=elem
    return norma

def leer_numeros_como_columna(archivo):
    # Cargar el archivo de texto y convertirlo en un array de numpy
    datos = np.loadtxt(archivo)  # Lee los números como un array unidimensional
    columna = datos[:, np.newaxis]  # Convertir a una columna (array de columna)
    return columna