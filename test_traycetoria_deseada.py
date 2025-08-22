import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros
value = 10  # Ajusta el valor según sea necesario
t_s = 1/30  # Paso de tiempo para el cálculo

# Tiempo
t = np.arange(0, 10, t_s)  # Rango de tiempo de 0 a 10 con pasos de t_s

# Definición de las funciones
xd = lambda t: 4 * np.sin(value * 0.04 * t) + 3
yd = lambda t: 4 * np.sin(value * 0.08 * t)

# Calcular las funciones
hxd = xd(t)
hyd = yd(t)

# Calcular las derivadas para obtener el ángulo de inclinación
dxdt = np.gradient(hxd, t_s)
dydt = np.gradient(hyd, t_s)
theta = np.arctan2(dydt, dxdt)  # Ángulo de inclinación en radianes

# Función de rotación
def rotate(x, y, angle):
    """Rote los puntos (x, y) en un ángulo dado (en radianes)."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y
    return x_rot, y_rot

# Almacenar las posiciones rotadas
rotated_positions = []

# Arrays para almacenar las nuevas trayectorias rotadas
hxd_rot = np.zeros_like(hxd)
hyd_rot = np.zeros_like(hyd)

for i in range(len(t)):
    angle = theta[i]
    x_rot, y_rot = rotate(hxd[:i+1], hyd[:i+1], -angle)
    
    # Mantener el punto actual en el origen (0,0)
    x_translated = x_rot# - x_rot[-1]
    y_translated = y_rot# - y_rot[-1]
    
    # Guardar las últimas posiciones rotadas
    hxd_rot[i] = x_translated[-1]
    hyd_rot[i] = y_translated[-1]
    
    rotated_positions.append((x_translated, y_translated))

# Convertir a arrays para la animación
rotated_positions = np.array(rotated_positions, dtype=object)

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2)
point, = ax.plot([], [], 'ro')  # Para marcar el punto actual
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('xd(t)')
ax.set_ylabel('yd(t)')
ax.set_title('Animación de la Trayectoria en el Plano XY (Rotada y Centrada en (0,0))')
ax.grid(True)

# Inicialización de la animación
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# Función de actualización para la animación
def update(frame):
    if frame >= len(rotated_positions):
        return line, point
    
    # Obtener las posiciones rotadas para el instante actual
    x_translated, y_translated = rotated_positions[frame]
    
    # Actualizar los datos de la trayectoria en la animación
    line.set_data(x_translated, y_translated)
    point.set_data(0, 0)  # Mantener el punto en el origen
    
    return line, point

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=len(rotated_positions), init_func=init, blit=True, interval=1000 * t_s * 10)

# Guardar la animación como archivo MP4
ani.save('trayectoria_animacion_origen.mp4', writer='ffmpeg', fps=30)

# Mostrar la animación (opcional, si se desea ver antes de guardar)
plt.show()

# Guardar las trayectorias rotadas en un archivo
np.savetxt('hxd_rot.txt', hxd_rot)
np.savetxt('hyd_rot.txt', hyd_rot)
