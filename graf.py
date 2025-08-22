import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_triangle(x, xref, save_filename):
    # Extraer los estados de posición y orientación de x
    y_positions = x[0, :]
    z_positions = x[1, :]
    orientations = x[2, :]

    # Obtener el número de cuadros o instantes de tiempo
    num_frames = x.shape[1]

    # Crear la figura y el eje
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.0, 10.0)

    # Inicializar el triángulo como un triángulo vacío
    triangle = plt.Polygon([[0, 0], [-0.5, 1], [0.5, 1]], closed=True, color='r')
    ax.add_patch(triangle)
    xref_line, = ax.plot([], [], 'b--')
    x_line, = ax.plot([], [], 'g-')

    # Función de animación que actualiza la posición y orientación del triángulo en cada cuadro
    def animate(i):
        # Calcular el ángulo de rotación basado en la tercera fila de x
        angle = orientations[i]

        # Calcular las coordenadas del triángulo en su posición original
        original_coords = np.array([[0, -0.55, 0.55],
                                    [0.15, 0, 0]])

        # Obtener la matriz de transformación de rotación
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        # Aplicar la rotación a las coordenadas del triángulo
        rotated_coords = rotation_matrix @ original_coords

        # Trasladar las coordenadas del triángulo a la posición correcta
        x_center = y_positions[i]
        y_center = z_positions[i]
        translated_coords = rotated_coords + np.array([[x_center], [y_center]])

        # Actualizar las coordenadas del triángulo
        triangle.set_xy(translated_coords.T)

        # Actualizar las coordenadas de xref en el triángulo
        xref_line.set_data(xref[0, :i+1], xref[1, :i+1])

        # Actualizar las coordenadas de x en el triángulo
        x_line.set_data(y_positions[:i+1], z_positions[:i+1])

        return triangle, xref_line, x_line

    # Crear la animación
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=0.5*100)

    # Guardar la animación en un archivo GIF usando imageio
    #writer = animation.ImageMagickFileWriter(fps=10)
    #anim.save(save_filename, writer=writer)
    

    # Guardar la animación en un archivo MP4
    
    anim.save(save_filename, writer='ffmpeg', codec='h264', fps=10)



