import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def quaternion_to_rotation_matrix(q):
    """Convierte un cuaternión [q0,q1,q2,q3] (w,x,y,z) a matriz de rotación 3x3."""
    w, x, y, z = q
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2*(y**2 + z**2)
    R[0, 1] = 2*(x*y - w*z)
    R[0, 2] = 2*(x*z + w*y)

    R[1, 0] = 2*(x*y + w*z)
    R[1, 1] = 1 - 2*(x**2 + z**2)
    R[1, 2] = 2*(y*z - w*x)

    R[2, 0] = 2*(x*z - w*y)
    R[2, 1] = 2*(y*z + w*x)
    R[2, 2] = 1 - 2*(x**2 + y**2)
    return R

def animate_SUV_pista(x, xref, left_poses, right_poses, save_filename, obstacles=None):
    """
    x: estados simulados (incluye posiciones y cuaterniones)
    xref: referencia
    left_poses, right_poses: puntos laterales calculados
    save_filename: nombre de archivo mp4
    obstacles: lista de (x, y, r) de obstáculos
    """

    # Extraer posiciones desde estados
    y_positions = x[0, :]   # eje X del mundo
    z_positions = x[1, :]   # eje Y del mundo
    quaternions = x[2:6, :] # q0..q3

    num_frames = x.shape[1]

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-7.5, 11)
    ax.set_ylim(-7.5, 7.5)

    # Dibujar obstáculos como círculos (si existen)
    circles = []
    if obstacles is not None:
        for j, (ox, oy, r) in enumerate(obstacles):
            circ = plt.Circle((ox, oy), r, color='purple', alpha=0.3, label="Obstáculo" if j == 0 else "")
            ax.add_patch(circ)
            circles.append(circ)

    # Triángulo inicial (coordenadas locales)
    triangle = plt.Polygon([[0, 0], [-0.5, 1], [0.5, 1]], closed=True, color='r')
    ax.add_patch(triangle)

    # Trayectorias
    xref_line, = ax.plot([], [], 'b--', label="Referencia")
    x_line, = ax.plot([], [], 'g-', label="Trayectoria real")

    # Puntos laterales
    left_points, = ax.plot([], [], 'bo', markersize=3, label='Puntos izquierdos')
    right_points, = ax.plot([], [], 'ro', markersize=3, label='Puntos derechos')

    ax.legend()

    def animate(i):
        # Cuaternión actual
        q = quaternions[:, i]
        R = quaternion_to_rotation_matrix(q)

        # Parte XY (2x2)
        R_2d = R[0:2, 0:2]

        # Coordenadas locales del barco
        original_coords = np.array([
            [0.6, -0.3, -0.3],
            [0.0,  0.2, -0.2]
        ])

        # Rotar y trasladar
        rotated_coords = R_2d @ original_coords
        translated_coords = rotated_coords + np.array([[y_positions[i]], [z_positions[i]]])

        # Actualizar triángulo
        triangle.set_xy(translated_coords.T)

        # Trayectorias
        xref_line.set_data(xref[0, :i+1], xref[1, :i+1])
        x_line.set_data(y_positions[:i+1], z_positions[:i+1])

        # Puntos laterales
        left_pos = left_poses[:, :, i]
        right_pos = right_poses[:, :, i]
        left_points.set_data(left_pos[:, 0], left_pos[:, 1])
        right_points.set_data(right_pos[:, 0], right_pos[:, 1])

        return triangle, xref_line, x_line, left_points, right_points, *circles

    # Crear animación
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    anim.save(save_filename, writer='ffmpeg', codec='h264', fps=10)

    anim.save("animation_MPCC_SUV.gif", writer="pillow", fps=10)


    return fig
