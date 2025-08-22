import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# -----------------------------
# 1. Definir waypoints y orientación deseada
# -----------------------------
points = np.array([
    [ 0.0,  0.0],
    [-2.5,  6.0],
    [ 3.5,  0.0],
    [-5.0, -6.0],
    [ 8.0, -4.0],
    [ 8.0,  6.0],
    [ 0.0,  0.0],
    [ 5,  5],
    [ 1,  -3]
], dtype=float)

# Orientaciones como tangentes
dirs = np.diff(points, axis=0)
orientations = np.arctan2(dirs[:,1], dirs[:,0])
orientations = np.append(orientations, orientations[-1])

# -----------------------------
# 2. Parámetros
# -----------------------------
freq = 30
dt = 1.0 / freq 
steps_per_segment = 2*freq  # pasos entre waypoints (asegúrate que sea suficiente)
N = steps_per_segment * (len(points)-1)  # horizonte global
n_states = 4
n_controls = 2

# Modelo masa puntual
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
# 3. Variables y costo
# -----------------------------
U = ca.MX.sym('U', n_controls, N)
X = ca.MX.sym('X', n_states, N+1)

Qth = 5
Ra = 0.1
Rw = 0.1
Qv = 2.0


cost = 0
g = []

# Estado inicial
x0 = np.array([points[0,0], points[0,1], 1.0, orientations[0]])
g.append(X[:,0] - x0)


Qw_smooth = 30
Qtheta_smooth = 30  # peso para suavizar cambios de orientación

# Dinámica y costo
for k in range(N):
    st = X[:,k]
    con = U[:,k]

    seg_idx = min(k//steps_per_segment, len(orientations)-1)
    cost += Qth*((st[3]-orientations[seg_idx])**2) \
          + Ra*(con[0]**2) + Rw*(con[1]**2)
    

    # Suavizado de control de giro
    if k > 0:
        cost += Qw_smooth * (U[1, k] - U[1, k-1])**2



    st_next = X[:,k] + dt*f(st, con)
    g.append(X[:,k+1] - st_next)

# -----------------------------

# -----------------------------
for i, wp in enumerate(points):
    idx = i * steps_per_segment 
    g.append(X[0, idx] - wp[0])            # x = x_wp
    g.append(X[1, idx] - wp[1])            # y = y_wp
    g.append(X[3, idx] - orientations[i])  # theta = theta_wp







        

# -----------------------------
# 5. Resolver NLP
# -----------------------------
OPT_variables = ca.vertcat(X.reshape((-1,1)), U.reshape((-1,1)))
g = ca.vertcat(*g)

nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g}
opts = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.tol': 1e-9,
    'ipopt.constr_viol_tol': 1e-9
}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Bounds
lbx = []
ubx = []
a_max = 25.0
omega_max = 15.0

for _ in range(N+1):
    lbx.extend([-ca.inf, -ca.inf, 0, -ca.inf])  # x, y, v >= 0, theta libre
    ubx.extend([ ca.inf,  ca.inf, 12,  ca.inf])
for _ in range(N):

    lbx.extend([-a_max, -omega_max])
    ubx.extend([ a_max,  omega_max])


# Igualdades exactas
lbg = [0]*(n_states*(N+1))  # dinámica
ubg = [0]*(n_states*(N+1))
lbg += [0]*(len(points)*3)  # x, y, theta
ubg += [0]*(len(points)*3)

sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
sol_x = sol['x'].full().flatten()

# -----------------------------
# 6. Extraer trayectoria
# -----------------------------
X_opt = sol_x[:n_states*(N+1)].reshape((N+1, n_states))
# Calcular tiempo total para llegar al final
T_total = N * dt
print(f"\nTiempo total estimado para llegar al último waypoint: {T_total:.2f} segundos")


# -----------------------------
# 7. Verificación en consola
# -----------------------------
print("\nVerificación de waypoints alcanzados:")
for i, wp in enumerate(points):
    idx = i * steps_per_segment
    x_val = X_opt[idx,0]
    y_val = X_opt[idx,1]
    th_val = X_opt[idx,3]
    print(f"WP {i}: ({x_val:.4f}, {y_val:.4f}), θ={th_val:.4f} rad | "
          f"objetivo=({wp[0]:.4f}, {wp[1]:.4f}), θ_obj={orientations[i]:.4f}")

# -----------------------------
# 8. Graficar
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(points[:,0], points[:,1], 'ro', label='Waypoints')
for i, ori in enumerate(orientations):
    plt.arrow(points[i,0], points[i,1],
              0.5*np.cos(ori), 0.5*np.sin(ori),
              head_width=0.2, color='r')
plt.plot(X_opt[:,0], X_opt[:,1], 'b-', label='Trayectoria óptima (pos+orientación)')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 8. Graficar trayectoria
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(points[:,0], points[:,1], 'ro', label='Waypoints')
for i, ori in enumerate(orientations):
    plt.arrow(points[i,0], points[i,1],
              0.5*np.cos(ori), 0.5*np.sin(ori),
              head_width=0.2, color='r')
plt.plot(X_opt[:,0], X_opt[:,1], 'b-', label='Trayectoria óptima (pos+orientación)')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 9. Graficar velocidades
# -----------------------------
time_axis = np.arange(0, (N+1)*dt, dt)

plt.figure(figsize=(8,4))
plt.plot(time_axis, X_opt[:,2], 'g-', linewidth=2)
plt.title('Perfil de velocidad en la trayectoria')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad [m/s]')
plt.grid(True)
plt.show()


# Extraer aceleraciones y omegas óptimos
U_opt = sol_x[n_states*(N+1):].reshape((N, n_controls))

# Graficar posición
plt.figure(figsize=(10,4))
plt.plot(time_axis, X_opt[:,0], label='x')
plt.plot(time_axis, X_opt[:,1], label='y')
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición [m]')
plt.title('Posiciones')
plt.legend()
plt.grid(True)
plt.show()

# Graficar velocidad y theta
plt.figure(figsize=(10,4))
plt.plot(time_axis, X_opt[:,2], label='Velocidad')
plt.plot(time_axis, X_opt[:,3], label='Theta')
plt.xlabel('Tiempo [s]')
plt.ylabel('Velocidad [m/s] / Theta [rad]')
plt.title('Velocidad y Orientación')
plt.legend()
plt.grid(True)
plt.show()

# Graficar aceleraciones y omegas
time_axis_u = time_axis[:-1] 
plt.figure(figsize=(10,4))
plt.plot(time_axis_u, U_opt[:,0], label='Aceleración a')
plt.plot(time_axis_u, U_opt[:,1], label='Omega')
plt.xlabel('Tiempo [s]')
plt.ylabel('Control')
plt.title('Controles óptimos')
plt.legend()
plt.grid(True)
plt.show()


# Construir xref para MPCC (igual que antes)
# =========================
n_states = 8
xd = np.zeros((n_states, X_opt[:,1].shape[0]))
xd[0, :] = X_opt[:,0] # X
xd[1, :] = X_opt[:,1]  # Y
xd[3, :] = X_opt[:,0]  # Vx
xd[4, :] = X_opt[:,1] # Vy
np.save("xref_optimo.npy", xd)
print(f"xref guardado como 'xref_optimo.npy' con forma {xd.shape}")