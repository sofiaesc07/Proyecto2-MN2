import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la funcion
#def funcion(x):
 #   total=0
  #  for i in range(len(x)):
   #     total+=x[i]**2
    #return total

#def funcion(x):
 #   A=10
  #  total=A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])
   # return total

def funcion(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


# Definir el algoritmo
def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Inciar las particulas y velocidades
    particles = np.random.uniform(-10, 10, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Mejor posición y ajuste
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterar a través del número especificado de iteraciones, actualizando la velocidad y posición de cada partícula en cada iteración
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        particles += velocities

        # Evaluar el ajuste de las particulas
        fitness_values = np.array([cost_func(p) for p in particles])

        # Actualizar las mejores posiciones y valores de aptitud
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    return swarm_best_position, swarm_best_fitness

dim = 2
solution, fitness = pso(funcion, dim=dim)
print('Solucion:', solution)
print('Ajuste:', fitness)

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = funcion([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter(solution[0], solution[1], fitness, color='red')
plt.show()

# Crear un gráfico de contorno en lugar de una superficie 3D
fig, ax = plt.subplots()
contour = ax.contour(X, Y, Z, cmap='viridis')

# Añadir un punto rojo para la solución encontrada por PSO
ax.scatter(solution[0], solution[1], color='red', marker='x', label='Solución')

# Etiquetas y título
ax.set_xlabel('x')
ax.set_ylabel('y')

# Mostrar la barra de color
plt.colorbar(contour)

# Mostrar leyenda
ax.legend()

# Mostrar el gráfico
plt.show()