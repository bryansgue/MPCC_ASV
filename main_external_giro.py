from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import  dot
from casadi import nlpsol
from casadi import sumsqr
from casadi import power
from casadi import diff
from fancy_plots import plot_pose, fancy_plots_2, fancy_plots_1
from graf_2 import animate_triangle_pista
from graf import animate_triangle
from plot_states import plot_states
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
#from graf import animate_triangle

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system
    m = 1.0  
    g = 9.81 
    Ixx = 0.02 

    # set up states
    hy = MX.sym('hy')
    hz = MX.sym('hz')
    psi = MX.sym('psi')
    hy_p = MX.sym('hy_p')
    hz_p = MX.sym('hz_p')
    psi_p = MX.sym('psi_p')
    x = vertcat(hy, hz, psi, hy_p, hz_p, psi_p)

    hy_pp = MX.sym('hy_pp')
    hz_pp = MX.sym('hz_pp')
    psi_pp = MX.sym('psi_pp')
    x_p = vertcat(hy_p, hz_p, psi_p, hy_pp, hz_pp, psi_pp)

    # set up states & controls
    F = MX.sym('F')
    T = MX.sym('T')
    u = vertcat(F, T)

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    psi_d = MX.sym('psi_d')

    nx_p_d = MX.sym('nx_d')
    ny_p_d = MX.sym('ny_d')
    psi_p_d = MX.sym('psi_d')


    F_d = MX.sym('F_d')
    T_d = MX.sym('T_d')

    xl_1 = MX.sym('xl_1')
    yl_1 = MX.sym('yl_1')
    xl_2 = MX.sym('xl_2')
    yl_2 = MX.sym('yl_2')
    xl_3 = MX.sym('xl_3')
    yl_3 = MX.sym('yl_3')

    
    p = vertcat(nx_d, ny_d, psi_d, nx_p_d, ny_p_d, psi_p_d, F_d , T_d)

    # Dynamic of the system
    R_system = MX.zeros(6, 2)
    R_system[3, 0] = (-1/m)*sin(psi)
    R_system[4, 0] = (1/m)*cos(psi)
    R_system[5, 0] = 0.0
    R_system[3, 1] = 0.0
    R_system[4, 1] = 0.0
    R_system[5, 1] = 1/Ixx

    h = MX.zeros(6,1)
    h[0, 0] = hy_p
    h[1, 0] = hz_p
    h[2, 0] = psi_p
    h[3, 0] = 0.0
    h[4, 0] = -g
    h[5, 0] = 0.0
    # dynamics
    f_expl = h + R_system@u
    f_system = Function('system',[x, u], [f_expl])

    # Acados Model
    f_impl = x_p - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = x_p
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system

def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x_next = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    return np.squeeze(x_next)

def create_ocp_solver_description(x0, N_horizon, t_horizon, bounded) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    F_max = bounded[0] 
    F_min = bounded[1] 
    T_max = bounded[2]
    T_min = bounded[3]

    model, f_system = f_system_model()
    ocp.model = model
    ocp.p = model.p
    
    
    # Calcula las dimensiones
    nx = model.x.shape[0]
    nu = model.u.shape[0]
    variables_adicionales = 0
    ny = nx + nu + variables_adicionales

    # set dimensions
    ocp.dims.N = N_horizon
    ocp.parameter_values = np.zeros(ny)

    # set cost
    Q_mat = 4 * np.diag([1, 1, 0, 0.0, 0.0, 0.0])  # [x,th,dx,dth]
    R_mat = 0*0.0000001 * np.diag([1,  1])

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    
    error_pose = ocp.p[0:6] - model.x[0:6]
    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u 
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose

    # Definir el tipo de restricciones
    ocp.constraints.constr_type = 'BGH'

    #Funcion de restriccion
    u = model.u
    x = model.x[0:3]

    constraints = vertcat(model.x[1])
    Dim_constraints = 1

    # Establecer la expresión de las restricciones en el modelo
    ocp.model.con_h_expr = constraints

    # Configurar los pesos de costo de las restricciones
    cost_weights = np.ones(Dim_constraints)
    ocp.cost.Zl = 1e3 * cost_weights
    ocp.cost.Zu = 1e3 * cost_weights
    ocp.cost.zl = cost_weights
    ocp.cost.zu = cost_weights

    # Configurar los límites inferior y superior de las restricciones
    ocp.constraints.lh = -1e9 * np.ones(Dim_constraints)  # límite inferior (min)
    ocp.constraints.uh = 20 * np.ones(Dim_constraints)   # límite superior (max)

    # Configurar los índices de las restricciones
    ocp.constraints.idxsh = np.arange(Dim_constraints)

    
    ocp.constraints.x0 = x0

    # set constraints
    ocp.constraints.lbu = np.array([F_min, T_min])
    ocp.constraints.ubu = np.array([F_max, T_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    #ocp.solver_options.sim_method_num_stages = 4
    #ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.levenberg_marquardt = 1e-2

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def calculate_unit_normals(t, xref):
    dx_dt = np.gradient(xref[0, :], t)
    dy_dt = np.gradient(xref[1, :], t)
    tangent_x = dx_dt / np.sqrt(dx_dt**2 + dy_dt**2)
    tangent_y = dy_dt / np.sqrt(dx_dt**2 + dy_dt**2)
    normal_x = -tangent_y
    normal_y = tangent_x
    return normal_x, normal_y

# Función de rotación
def rotate(x, y, angle):
    """Rote los puntos (x, y) en un ángulo dado (en radianes)."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * x - sin_angle * y
    y_rot = sin_angle * x + cos_angle * y
    return x_rot, y_rot


def main():

    plt.figure(figsize=(10, 5))

    t_final = 20
    # Sample time
    frecuencia = 30
    t_s = 1/frecuencia
 
    Horizont = 30
    t_prediction = Horizont/frecuencia

    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0] #  devuelve la longitud o cantidad de elementos en N.
    print(N_prediction)

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)
    x = np.zeros((6, t.shape[0]+1-N_prediction), dtype = np.double)
    u_control = np.zeros((2, t.shape[0]-N_prediction), dtype = np.double)

    g = 9.8
    m0 = 1.0
    I_xx = 0.02
    L = [g, m0, I_xx]

    # Definición de las funciones
    value = 5
    xd = lambda t: 4 * np.sin(value * 0.04 * t) 
    yd = lambda t: 4 * np.sin(value * 0.08 * t)

    # Calcular las funciones
    hxd = xd(t)
    hyd = yd(t)

    # Calcular las derivadas para obtener el ángulo de inclinación
    dxdt = np.gradient(hxd, t_s)
    dydt = np.gradient(hyd, t_s)
    theta = np.arctan2(dydt, dxdt)  # Ángulo de inclinación en radianes
    # Reference Signal of the system
    xref = np.zeros((8, t.shape[0]), dtype = np.double)
    xref[0, :] =  hxd
    xref[1, :] = hyd
    xref[2,:] = 45*(np.pi)/180
    xref[3,:] = 0.0 
    xref[4,:] = 0.0
    xref[5,:] = 0.0

    # Arrays para almacenar las nuevas trayectorias rotadas
    hxd_rot = np.zeros_like(hxd)
    hyd_rot = np.zeros_like(hyd)

    # Almacenar las posiciones rotadas
    rotated_positions = []

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

    #   Convertir a arrays para la animación
    rotated_positions = np.array(rotated_positions, dtype=object)

    xref[0, :] =  hxd_rot   
    xref[1, :] =  hxd_rot
    
    model, f = f_system_model()

    # Maximiun Values
    f_max = 3*m0*g
    f_min = 0

    t_max = 0.5
    t_min = -t_max 

    bounded = [f_max, f_min, t_max, t_min]

    # Optimization Solver
    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, bounded)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    # Init states system
    # Dimentions System
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
  
    print("AQUI TA")
    for k in range(0, t.shape[0]-N_prediction):

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # update yref
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "p", yref)
        
        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p", yref_N)

        # Get Computational Time
        tic = time.time()
        # solve ocp
        status = acados_ocp_solver.solve()

        toc = time.time()- tic
        #print(toc)

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        # System Evolution
        x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
        delta_t[:, k] = toc


    fig3 = animate_triangle(x[:3, :], xref[:2, :], 'animation.mp4')


    # Crear la animación
    ani = animation.FuncAnimation(fig, update, frames=len(rotated_positions), init_func=init, blit=True, interval=1000 * t_s * 10)

   # plot_states(x[:3, :], xref[:3, :], 'states_plot.png')

     
    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



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

if __name__ == '__main__':
    main()
