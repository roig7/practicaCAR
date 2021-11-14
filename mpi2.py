from mpi4py import MPI 
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
from skimage.io import imread
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
import math
import time


#Inicio del tiempo
start = time.perf_counter()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# Metemos todo lo que no se queire paralelizar dentro del rank =0 
if rank==0:
    img = plt.imread("s.png")
    gray = rgb2gray(img)
    filtro = gaussian_filter(gray,sigma=1)
    # Encuentra el gradiente en la dirección X
    grad_x = cv.Sobel(filtro, cv.CV_16SC1, 1, 0)
    # Encuentra el gradiente en la dirección y
    grad_y = cv.Sobel(filtro, cv.CV_16SC1, 0, 1)
    # Convertir el valor del gradiente a 8 bits
    x_grad = cv.convertScaleAbs(grad_x)
    y_grad = cv.convertScaleAbs(grad_y)
    # Combina dos gradientes
    combina_grad = cv.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)
    combina_grad=np.array(combina_grad,dtype=float)
else:#Metemos las variables que esten por encima que se usan en el bloque se vaya a paralelizar.
    combina_grad = None

#Broadcast 
combina_grad = comm.bcast(combina_grad, root=0)

# Combina dos gradientes
filas, columnas = combina_grad.shape

imagen_final = np.zeros([filas, columnas])               
# Definir umbral alto y bajo
bajo = 0.2 * np.max(combina_grad)
alto = 0.3 * np.max(combina_grad)

filasproc=int(combina_grad.shape[0]/size)
filasresto=int(combina_grad.shape[0]%size)

arrayvacio=[]
for i in range(size):
    arrayvacio.append(filasproc*columnas)

arrayvacio[0]+=filasresto*columnas
# Divide the data among processes pero solo los datos importantes
data=np.zeros((filasproc, columnas),dtype=float)
if rank==0:
    data=np.zeros((filasproc+filasresto, columnas),dtype=float)

#Scatter 
comm.Scatterv(sendbuf=(combina_grad,arrayvacio),recvbuf=data, root=0)

#For que se paraleliza
imagen_final_data=np.zeros([data.shape[0], data.shape[1]])  
for i in range(1, data.shape[0]-1):
    for j in range(1, data.shape[1]-1):
        if (data[i, j] < bajo):
            imagen_final_data[i, j] = 0
        elif (data[i, j] > alto):
            imagen_final_data[i, j] = 1
        elif ((data[i-1, j-1:j+1] < alto).any() or (data[i+1, j-1:j+1]).any() 
            or (data[i, [j-1, j+1]] < alto).any()):
            imagen_final_data[i, j] = 1



#gather de la info de todos los procesos
comm.Gatherv(sendbuf=imagen_final_data,recvbuf=(imagen_final,arrayvacio), root=0)

#El proceso 0 es el unico proceso que imprime la imagen
if (rank==0):
    plt.imshow(imagen_final,cmap='gray')
    plt.show()
    print(time.perf_counter()-start)
