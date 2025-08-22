import numpy as np
import matplotlib.pyplot as plt

def plot_states(x, xref, save_filename):
    # Extraer los primeros tres estados de x
    y_positions = x[0, :]
    z_positions = x[1, :]
    orientations = x[2, :]

    # Crear la figura y los subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Configurar los límites de los ejes
    axs[0].set_xlim(0, x.shape[1])
    axs[0].set_ylim(np.min(y_positions), np.max(y_positions))
    axs[1].set_xlim(0, x.shape[1])
    axs[1].set_ylim(np.min(z_positions), np.max(z_positions))
    axs[2].set_xlim(0, x.shape[1])
    axs[2].set_ylim(np.min(orientations), np.max(orientations))

    # Graficar y_positions
    axs[0].plot(y_positions, 'b-', label='y_positions')
    axs[0].plot(xref[0, :], 'r--', label='xref_y_positions')
    axs[0].legend()

    # Graficar z_positions
    axs[1].plot(z_positions, 'g-', label='z_positions')
    axs[1].plot(xref[1, :], 'r--', label='xref_z_positions')
    axs[1].legend()

    # Graficar orientations
    axs[2].plot(orientations, 'm-', label='orientations')
    axs[2].plot(xref[2, :], 'r--', label='xref_orientations')
    axs[2].legend()

    # Ajustar los espacios entre los subplots
    plt.tight_layout()

    # Guardar la gráfica en un archivo PNG
    plt.savefig(save_filename)

    # Mostrar el gráfico
    plt.show()