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
    dimA=len(A)
    Kinv=np.zeros((dimA,dimA))     # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    cords=0
    for fila in A:
        cant_conexiones=0
        for i in range (dimA-1):
            cant_conexiones+=fila[i]
        Kinv[cords][cords]=1/cant_conexiones
    C = (np.transpose(A)) @ Kinv # Calcula C multiplicando Kinv y A
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
    D = D.copy()
    F = 1/D
    # F = calcular_inversa(D)

    A=construye_adyacencia(D,3)
    matriz_continua=np.zeros(len(A),len(A))
    for i in range (len(A)):
        for j in range (len(A)):
            if (A[i][j]!=1):
                matriz_continua[i][j]= A[i][j] + F[i][j]
            else:
                matriz_continua[i][j]=1
    C = calcula_matriz_C(matriz_continua)

    """np.fill_diagonal(F,0)
    dimF = len(F)
    Kinv=np.zeros((dimF,dimF)) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    cant_con = np.sum(F, axis = 0) # Sumo por columnas
    Kinv = np.diag(1/cant_con)"""
    
    # cords=0
    # """for fila in F:
    #     cant_conexiones=0
    #     for i in range (dimF-1):
    #         cant_conexiones+=fila[i]
    #     Kinv[cords][cords]=1/cant_conexiones """
    
    #C = F @ Kinv # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        # Sumamos las matrices de transición para cada cantidad de pasos
        if i == 0:
            continue
        elif (i==1):
            B += C
        else:    
            C_potencia = C_potencia @ C  # Multiplicamos una vez por C cada vuelta
            B += C_potencia  # Sumamos la nueva potencia
    return B

