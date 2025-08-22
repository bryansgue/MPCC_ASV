import numpy as np
import matplotlib.pyplot as plt

def calculate_xref(t):
    value = 5
    xref = np.zeros((2, len(t)))
    xref[0, :] = 4 * np.sin(value*0.03*t) + 3
    xref[1, :] = 4 * np.sin(value*0.06*t)
    return xref

def calculate_unit_normals(t, xref):
    dx_dt = np.gradient(xref[0, :], t)
    dy_dt = np.gradient(xref[1, :], t)
    tangent_x = dx_dt / np.sqrt(dx_dt**2 + dy_dt**2)
    tangent_y = dy_dt / np.sqrt(dx_dt**2 + dy_dt**2)
    normal_x = -tangent_y
    normal_y = tangent_x
    return normal_x, normal_y

def displace_points_along_normal(x, y, normal_x, normal_y, displacement):
    x_prime = x + displacement * normal_x
    y_prime = y + displacement * normal_y
    return x_prime, y_prime

t = np.linspace(0, 30, 100)
xref = calculate_xref(t)
normal_x, normal_y = calculate_unit_normals(t, xref)

track_width = 0.8

plt.figure(figsize=(10, 5))

for i in range(4, len(t), 4):
    left_displacement = -0.5 * track_width
    right_displacement = 0.5 * track_width

    left_x, left_y = displace_points_along_normal(xref[0, i-4:i], xref[1, i-4:i], normal_x[i-4:i], normal_y[i-4:i], left_displacement)
    right_x, right_y = displace_points_along_normal(xref[0, i-4:i], xref[1, i-4:i], normal_x[i-4:i], normal_y[i-4:i], right_displacement)

    # Interpolar un polinomio de grado 3 para el lado izquierdo
    poly_func_left = np.poly1d(np.polyfit(left_x, left_y, 2))

    # Interpolar un polinomio de grado 3 para el lado derecho
    poly_func_right = np.poly1d(np.polyfit(right_x, right_y, 2))

    # Evaluar las funciones polinómicas en un rango de valores x para obtener curvas suavizadas
    x_range_left = np.linspace(min(left_x), max(left_x), 100)
    y_interp_left = poly_func_left(x_range_left)

    x_range_right = np.linspace(min(right_x), max(right_x), 100)
    y_interp_right = poly_func_right(x_range_right)

    # Graficar las curvas interpoladas
    plt.clf()
    plt.plot(xref[0, :], xref[1, :], label='Central Track', color='black')
    plt.plot(left_x, left_y, 'o', label='Puntos originales (izquierdo)', color='blue')
    plt.plot(x_range_left, y_interp_left, label='Curva interpolada (izquierdo)', color='blue')

    plt.plot(right_x, right_y, 'o', label='Puntos originales (derecho)', color='red')
    plt.plot(x_range_right, y_interp_right, label='Curva interpolada (derecho)', color='red')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.pause(0.5)  # Pausa para visualizar cada iteración

plt.show()
