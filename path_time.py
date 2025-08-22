import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import os
from scipy.io import savemat


# -----------------------------
# 1. Definir waypoints y orientación deseada
# -----------------------------
points = np.array([
    [ 0.0,  0.0],
    [-2.5,  6.0],
    [ 3.5,  0.0],
    [-5.0, -6.0],
    [0, -4.0],
    [ 8.0, -4.0],
    [ 8.0,  6.0],
    [ 10.0,  0.0],
    [ 10,  -6]
], dtype=float)

# Orientaciones como tangentes
dirs = np.diff(points, axis=0)
orientations = np.arctan2(dirs[:,1], dirs[:,0])
orientations = np.append(orientations, orientations[-1])

# -----------------------------
# 2. Parámetros
# -----------------------------
freq = 30  # resolución temporal (steps/seg)
vel_est = 1.0  # velocidad estimada para cálculo inicial
distancias = np.linalg.norm(np.diff(points, axis=0), axis=1)
dist_total = np.sum(distancias)

# Horizonte N (fijo en pasos, pero dt variable)
tiempo_total_est = dist_total / vel_est
N = int(np.ceil(tiempo_total_est * freq))

n_states = 4
n_controls = 2

# Calcular índice estimado de cada waypoint
pasos_por_metro = N / dist_total
idx_wp_estimado = [0]
dist_acum = 0
for d in distancias:
    dist_acum += d
    idx_wp_estimado.append(int(dist_acum * pasos_por_metro))

# -----------------------------
# 3. Modelo masa puntual
# -----------------------------
x = ca.MX.sym('x')
y = ca.MX.sym('y')
v = ca.MX.sym('v')
theta = ca.MX.sym('theta')
states = ca.vertcat(x, y, v, theta)

a = ca.MX.sym('a')
omega = ca.MX.sym('omega')
controls = ca.vertcat(a, omega)

rhs = ca.vertcat(
    v*ca.cos(theta),
    v*ca.sin(theta),
    a,
    omega
)
f = ca.Function('f', [states, controls], [rhs])

# -----------------------------
# 4. Variables de optimización
# -----------------------------
U = ca.MX.sym('U', n_controls, N)
X = ca.MX.sym('X', n_states, N+1)
T_total_var = ca.MX.sym('T_total')  # tiempo total como variable
dt_var = T_total_var / N            # paso de integración variable

# Pesos de costo
Qth = 5
Ra = 0.01
Rw = 0.1
Qw_smooth = 10

Qx_wp = 50
Qy_wp = 50
Qtheta_wp = 150

w_T = 50  # peso para minimizar tiempo total

cost = 0
g = []

# Estado inicial duro
x0_val = np.array([points[0,0], points[0,1], vel_est, orientations[0]])
g.append(X[:,0] - x0_val)

# -----------------------------
# 5. Dinámica y costo
# -----------------------------
for k in range(N):
    st = X[:,k]
    con = U[:,k]

    cost += Ra * (con[0]**2) + Rw * (con[1]**2)
    if k > 0:
        cost += Qw_smooth * (U[1,k] - U[1,k-1])**2

    st_next = X[:,k] + dt_var * f(st, con)
    g.append(X[:,k+1] - st_next)

# -----------------------------
# 6. Penalización suave en waypoints
# -----------------------------
for i, wp in enumerate(points):
    idx = min(idx_wp_estimado[i], N)
    cost += Qx_wp * ((X[0, idx] - wp[0])**2) \
          + Qy_wp * ((X[1, idx] - wp[1])**2) \
          + Qtheta_wp * ((X[3, idx] - orientations[i])**2)

# Penalizar tiempo total directamente
cost += w_T * T_total_var

# -----------------------------
# 7. Resolver NLP
# -----------------------------
OPT_variables = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)), T_total_var)
g = ca.vertcat(*g)

nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g}
opts = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.tol': 1e-8,
    'ipopt.constr_viol_tol': 1e-8
}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Bounds
lbx = []
ubx = []

# Estados
for _ in range(N+1):
    lbx.extend([-ca.inf, -ca.inf, 0, -ca.inf])
    ubx.extend([ ca.inf,  ca.inf, 12,  ca.inf])

# Controles
a_max = 25.0
omega_max = 15.0
for _ in range(N):
    lbx.extend([-a_max, -omega_max])
    ubx.extend([ a_max,  omega_max])

# Tiempo total
lbx.append(0.1)    # mínimo tiempo posible
ubx.append(200.0)  # máximo tiempo

# Restricciones
lbg = [0]*(n_states*(N+1))
ubg = [0]*(n_states*(N+1))

# Valor inicial
x0_guess = np.zeros(len(lbx))
x0_guess[-1] = tiempo_total_est

sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0_guess)
sol_x = sol['x'].full().flatten()

# -----------------------------
# 8. Extraer resultados
# -----------------------------
X_opt = sol_x[:n_states*(N+1)].reshape((N+1, n_states))
U_opt = sol_x[n_states*(N+1):-1].reshape((N, n_controls))
T_total_opt = sol_x[-1]

print(f"\nTiempo total óptimo: {T_total_opt:.2f} segundos")

print("\nDistancia y orientación respecto a waypoints:")
for i, wp in enumerate(points):
    idx = min(idx_wp_estimado[i], N)
    dx = X_opt[idx,0] - wp[0]
    dy = X_opt[idx,1] - wp[1]
    dth = X_opt[idx,3] - orientations[i]
    print(f"WP {i}: Δx={dx:.4f}, Δy={dy:.4f}, Δθ={dth:.4f}")

# -----------------------------
# 9. Graficar trayectoria con gates
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(points[:, 0], points[:, 1], 'ro', label='Waypoints')

gate_length = 1.0  # largo total de cada gate
for i, ori in enumerate(orientations):
    # Vector perpendicular a la orientación
    perp_angle = ori + np.pi/2
    dx = (gate_length/2) * np.cos(perp_angle)
    dy = (gate_length/2) * np.sin(perp_angle)

    # Línea del gate (centro en waypoint)
    plt.plot([points[i, 0] - dx, points[i, 0] + dx],
             [points[i, 1] - dy, points[i, 1] + dy],
             'k-', linewidth=2)

plt.plot(X_opt[:, 0], X_opt[:, 1], 'b-', label='Trayectoria óptima')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.title("0) Path Planning: Waypoints y Trayectoria Óptima")
plt.savefig("0_path_planning.png", dpi=300)
plt.show()

# -----------------------------
# 10. Graficar velocidad
# -----------------------------
time_axis = np.linspace(0, T_total_opt, N+1)
plt.figure(figsize=(8,4))
plt.plot(time_axis, X_opt[:,2], 'g-', linewidth=2)
plt.title('Perfil de velocidad')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad [m/s]')
plt.grid(True)
plt.show()

# -----------------------------
# 11. Graficar controles
# -----------------------------
time_axis_u = np.linspace(0, T_total_opt, N)
plt.figure(figsize=(10,4))
plt.plot(time_axis_u, U_opt[:,0], label='Aceleración a')
plt.plot(time_axis_u, U_opt[:,1], label='Omega')
plt.xlabel('Tiempo [s]')
plt.ylabel('Control')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 12. Guardar referencia
# -----------------------------
# Número de muestras reales según el tiempo total óptimo y la frecuencia
n_samples = int(np.ceil(T_total_opt * freq))

# Interpolamos X_opt para tener exactamente n_samples puntos
time_original = np.linspace(0, T_total_opt, N+1)
time_resampled = np.linspace(0, T_total_opt, n_samples)



X_resampled = np.zeros((n_samples, n_states))
for i in range(n_states):
    X_resampled[:, i] = np.interp(time_resampled, time_original, X_opt[:, i])

# Guardar referencia con tamaño correcto
n_states_ref = 8
xd = np.zeros((n_states_ref, n_samples))
xd[0, :] = X_resampled[:, 0]  # x
xd[1, :] = X_resampled[:, 1]  # y
xd[3, :] = X_resampled[:, 0]  # x repetido
xd[4, :] = X_resampled[:, 1]  # y repetido


# Redondear a 2 decimales y asegurar que sea float
T_total_opt = np.round(T_total_opt, 2).astype(float)

# Guardar

np.save("xref_optimo.npy", xd)
np.save("tref_optimo.npy", T_total_opt)

print(f"xref guardado como 'xref_optimo.npy' con forma {xd.shape}")
print(f"tiempo guardado como 'tref_optimo.npy' con forma {np.shape(T_total_opt)} y dtype {T_total_opt.dtype}")




# -----------------------------
# 13. Calcular velocidades en X y Y
# -----------------------------
vx = X_opt[:, 2] * np.cos(X_opt[:, 3])
vy = X_opt[:, 2] * np.sin(X_opt[:, 3])

# -----------------------------
# 14. Calcular aceleraciones en X y Y
# -----------------------------
ax = np.gradient(vx, time_axis)
ay = np.gradient(vy, time_axis)

# -----------------------------
# 15. Graficar en subplots
# -----------------------------
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Posiciones
axs[0].plot(time_axis, X_opt[:,0], label='Posición X')
axs[0].plot(time_axis, X_opt[:,1], label='Posición Y')
axs[0].set_ylabel('Posición [m]')
axs[0].grid(True)
axs[0].legend()
axs[0].set_title('Posición, Velocidad y Aceleración')

# Velocidades
axs[1].plot(time_axis, vx, label='Velocidad X')
axs[1].plot(time_axis, vy, label='Velocidad Y')
axs[1].set_ylabel('Velocidad [m/s]')
axs[1].grid(True)
axs[1].legend()

# Aceleraciones
axs[2].plot(time_axis, ax, label='Aceleración X')
axs[2].plot(time_axis, ay, label='Aceleración Y')
axs[2].set_xlabel('Tiempo [s]')
axs[2].set_ylabel('Aceleración [m/s²]')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()

# ============================
#  Definir tiempo uniforme (como en el otro programa)
# ============================
frec = 30  # Hz
t_final = float(T_total_opt)
t = np.linspace(0, t_final, int(np.ceil(t_final * frec)))

# Interpolar posiciones y velocidades en función de t
interp_x  = interp1d(time_original, X_opt[:,0], kind='cubic')
interp_y  = interp1d(time_original, X_opt[:,1], kind='cubic')
interp_v  = interp1d(time_original, X_opt[:,2], kind='cubic')
interp_th = interp1d(time_original, X_opt[:,3], kind='cubic')

xd_array = interp_x(t)
yd_array = interp_y(t)
v_array  = interp_v(t)
th_array = interp_th(t)

# Velocidades en XY
vx_array = v_array * np.cos(th_array)
vy_array = v_array * np.sin(th_array)

# Derivadas en XY
ax_array = np.gradient(vx_array, t)
ay_array = np.gradient(vy_array, t)

# ============================
#  Guardar resultados en .mat
# ============================
pwd = "/home/bryansgue/Doctoral_Research/Matlab/MPCC_USV"
if not os.path.exists(pwd):
    print(f"La ruta {pwd} no existe, se usará el directorio actual.")
    pwd = os.getcwd()

experiment_number = 1
name_file = "Results_PathPlanning" + str(experiment_number) + ".mat"

savemat(os.path.join(pwd, name_file), {
    'time': t,                   # tiempo único
    'x': xd_array,               # trayectoria x
    'y': yd_array,               # trayectoria y
    'v': v_array,                 # velocidad escalar
    'theta': th_array,           # orientación
    'vx': vx_array, 'vy': vy_array,
    'ax': ax_array, 'ay': ay_array,
    'points': points,
    'orientations': orientations
})
print(f"Datos de planificación guardados en {name_file}")