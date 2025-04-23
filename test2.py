import template_funciones as tf
import numpy as np
import scipy
import matplotlib.pyplot as pl
import pandas as pd
import geopandas as gpd

museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')

#matriz distancia y adyacencia

D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
m = 3 # Cantidad de links por nodo
A = tf.construye_adyacencia(D,m)
#test LU
test1=[[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]]
#       L,U=tf.calculaLU(np.asarray(test1))
#       print(L , U)

#test inversa

#test transpuesta

transpuesta=tf.calcular_transpuesta(D)
#print(transpuesta)

#test matriz A

#matrizA=tf.construye_adyacencia(D,2)
#print("La matriz A es" , A)

#test matriz C

#matrizC= tf.calcula_matriz_C(matrizA)
        #print(matrizC)

#test pagerank

#test matriz C continua
matrizC_cont=tf.calcula_matriz_C_continua(D)
#print(matrizC_cont)

B= tf.calcula_B(matrizC_cont,3)
#print(B)
L,U=tf.calculaLU(B)
#print(L,U)
# Leer el archivo y convertir los números en un array de columna
def leer_numeros_como_columna(archivo):
    # Cargar el archivo de texto y convertirlo en un array de numpy
    datos = np.loadtxt(archivo)  # Lee los números como un array unidimensional
    columna = datos[:, np.newaxis]  # Convertir a una columna (array de columna)
    return columna

# Llamamos a la función con el archivo de texto
archivo = "tp1/visitas.txt"
w = leer_numeros_como_columna(archivo)
#print(columna)

v=tf.calculaV(B,w)
print(v)

def calcNorma(v):
    norma=0
    for elem in v:
        norma+=elem
    return norma

normaV = calcNorma(v)
normaW = calcNorma(w)
print(normaV)
#print(np.linalg.norm(v,ord=1))
print(normaW)
#print(np.linalg.norm(w,ord=1))