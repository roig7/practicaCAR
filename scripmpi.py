from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # get your process ID
# cogemos los datos que se van a utilizar
if rank == 0: 
    filas, columnas = combina_grad.shape
	imagen_final = np.zeros([filas, columnas])               
	# Definir umbral alto y bajo
	bajo = 0.2 * np.max(combina_grad)
	alto = 0.3 * np.max(combina_grad)

# Divide the data among processes pero solo los datos importantes
data = comm.scatter(combina_grad, root=0)
# paso 4. Algoritmo de doble umbral para detectar y conectar bordes
#bucle a paralelizar
for i in range(1, filas-1):
    for j in range(1, columnas-1):
        if (combina_grad[i, j] < bajo):
            imagen_final[i, j] = 0
        elif (combina_grad[i, j] > alto):
            imagen_final[i, j] = 1
        elif ((combina_grad[i-1, j-1:j+1] < alto).any() or (combina_grad[i+1, j-1:j+1]).any() 
              or (combina_grad[i, [j-1, j+1]] < alto).any()):
            imagen_final[i, j] = 1
#gather de la info de todos los procesos
imagen_final = comm.gather(imagen_final,root=0)
