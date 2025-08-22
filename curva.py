import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def ajustar_polinomio_grado_3(x_data, y_data):
    # Definir variables simbólicas para los coeficientes del polinomio de grado 3
    a = ca.MX.sym('a', 4)

    # Modelo a ajustar: polinomio de grado 3: y = a3*x^3 + a2*x^2 + a1*x + a0
    model = a[3] * x_data**3 + a[2] * x_data**2 + a[1] * x_data + a[0]

    # Residuos (diferencia entre modelo y datos)
    residuals = model - y_data

    # Función de costo: mínimos cuadrados
    cost_function = ca.dot(residuals, residuals)

    # Crear un solver de optimización
    nlp = {'x': a, 'f': cost_function}
    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Resolver el problema de optimización
    initial_guess = [0.0, 0.0, 0.0, 0.0]  # Valores iniciales para los coeficientes del polinomio
    solution = solver(x0=initial_guess)

    # Obtener los valores ajustados de los coeficientes del polinomio
    a_sol = solution['x']
    
    return a_sol

# Datos de ejemplo (puntos x, y)
x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y_data = np.array([0.1, 3.9, 5.2, 6.8, 1.9])

# Ajustar polinomio
coeficientes_polinomio = ajustar_polinomio_grado_3(x_data, y_data)
print("Coeficientes del polinomio ajustado:", coeficientes_polinomio)

# Graficar los datos y la curva ajustada
x_plot = np.linspace(min(x_data), max(x_data), 100)
#y_plot = coeficientes_polinomio[3]*ca.power(x_plot, 3) + coeficientes_polinomio[2]*ca.power(x_plot, 2) + coeficientes_polinomio[1]*x_plot + coeficientes_polinomio[0]
y_plot = np.polyval(coeficientes_polinomio[::-1], x_plot)

plt.scatter(x_data, y_data, label='Datos')
plt.plot(x_plot, y_plot, color='red', label='Curva ajustada')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Ajuste de curva usando CasADi')
plt.grid(True)
plt.show()
