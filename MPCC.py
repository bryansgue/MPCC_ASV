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
from scipy.integrate import quad
from scipy.optimize import bisect
from casadi import dot, norm_2, mtimes, DM, SX, MX,  if_else

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

    el_x = MX.sym('el_x')
    el_y = MX.sym('el_y')
    ec_x = MX.sym('ec_x')
    ec_y = MX.sym('ec_y')
    
    theta_p = MX.sym('theta_p')

    p = vertcat(nx_d, ny_d, psi_d, nx_p_d, ny_p_d, psi_p_d, F_d , T_d , el_x, el_y, ec_x, ec_y, theta_p)

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
    variables_adicionales = 5
    ny = nx + nu + variables_adicionales

    # set dimensions
    ocp.dims.N = N_horizon
    ocp.parameter_values = np.zeros(ny)

    # set cost
    Q_mat = 1.5 * np.diag([1, 1])  # [x,th,dx,dth]
    R_mat = 0.00001 * np.diag([1,  1])

    # Define matrices de ganancia para los errores
    Q_el = 10 * np.eye(2)  # Ganancia para el error el (2x2)
    Q_ec = 10* np.eye(2)  # Ganancia para el error ec (2x2)
    Q_theta_p = 500  # Ganancia para theta_p (escalar)
    R_u = 0.01 * np.diag([1, 1])

    # Definir los errores como vectores
 

    # Definir variables simbólicas  
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    
    #ERROR DE POSICION
    sd = ocp.p[0:2]
    error_pose = sd - model.x[0:2]

    #ERROR DE ARRASTRE
    sd_p = ocp.p[3:5]
    tangent_normalized = sd_p / norm_2(sd_p)
    el = dot(tangent_normalized, error_pose) * tangent_normalized

    # ERROR DE CONTORNO
    I = MX.eye(2) 
    P_ec = I - tangent_normalized.T @ tangent_normalized
    ec = P_ec @ error_pose 

    # Define el costo externo considerando los errores como vectores
    error_pos = 0*error_pose.T @ Q_mat @error_pose 
    error_contorno = 1*ec.T @ Q_ec @ ec
    error_lag = 1*el.T @ Q_el @ el
    ocp.model.cost_expr_ext_cost = (error_contorno + error_lag + 1*model.u.T @ R_u @ model.u )#- Q_theta_p * theta_p**2 )
    ocp.model.cost_expr_ext_cost_e = (error_contorno + error_lag) # -  Q_theta_p * theta_p**2 )

        
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

def displace_points_along_normal(x, y, normal_x, normal_y, displacement):
    x_prime = x + displacement * normal_x
    y_prime = y + displacement * normal_y
    return x_prime, y_prime



# Definir el valor global
value = 21

def trayectoria(t):
    """ Crea y retorna las funciones para la trayectoria y sus derivadas. """
    def xd(t):
        return 4 * np.sin(value * 0.04 * t) + 1

    def yd(t):
        return 4 * np.sin(value * 0.08 * t)

    def zd(t):
        return 2 * np.sin(value * 0.08 * t) + 6

    def xd_p(t):
        return 4 * value * 0.04 * np.cos(value * 0.04 * t)

    def yd_p(t):
        return 4 * value * 0.08 * np.cos(value * 0.08 * t)

    def zd_p(t):
        return 2 * value * 0.08 * np.cos(value * 0.08 * t)

    return xd, yd, zd, xd_p, yd_p, zd_p

def r(t, xd, yd, zd):
    """ Devuelve el punto en la trayectoria para el parámetro t usando las funciones de trayectoria. """
    return np.array([xd(t), yd(t), zd(t)])

def r_prime(t, xd_p, yd_p, zd_p):
    """ Devuelve la derivada de la trayectoria en el parámetro t usando las derivadas de las funciones de trayectoria. """
    return np.array([xd_p(t), yd_p(t), zd_p(t)])

def integrand(t, xd_p, yd_p, zd_p):
    """ Devuelve la norma de la derivada de la trayectoria en el parámetro t. """
    return np.linalg.norm(r_prime(t, xd_p, yd_p, zd_p))

def arc_length(tk, t0=0, xd_p=None, yd_p=None, zd_p=None):
    """ Calcula la longitud de arco desde t0 hasta tk usando las derivadas de la trayectoria. """
    length, _ = quad(integrand, t0, tk, args=(xd_p, yd_p, zd_p))
    return length

def find_t_for_length(theta, t0=0, t_max=None, xd_p=None, yd_p=None, zd_p=None):
    """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
    func = lambda t: arc_length(t, t0, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p) - theta
    return bisect(func, t0, t_max)

def length_to_point(theta, t0=0, t_max=None, xd=None, yd=None, zd=None, xd_p=None, yd_p=None, zd_p=None):
    """ Convierte una longitud de arco theta a un punto en la trayectoria. """
    tk = find_t_for_length(theta, t0, t_max, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)
    return r(tk, xd, yd, zd)

def calculate_positions_in_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):
    """ Calcula los puntos en la trayectoria y la longitud de arco para cada instante en t_range. """
    positions = []
    arc_lengths = []
    for tk in t_range:
        theta = arc_length(tk, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)
        arc_lengths.append(theta)
        point = length_to_point(theta, t_max=t_max, xd=xd, yd=yd, zd=zd, xd_p=xd_p, yd_p=yd_p, zd_p=zd_p)
        positions.append(point)
    return np.array(arc_lengths), np.array(positions).T

def calculate_orthogonal_error(error_total, tangent):

    if np.linalg.norm(tangent) == 0:
        return error_total  # No hay tangente válida, devolver el error total
    # Matriz de proyección ortogonal
    I = np.eye(2)  # Matriz identidad en 3D
    P_ec = I - np.outer(tangent, tangent)
    # Aplicar la matriz de proyección para obtener el error ortogonal
    e_c = P_ec @ error_total
    return e_c

def main():

    plt.figure(figsize=(10, 5))
    # Initial Values System
    t_final = 15
    frec = 30
    t_s = 1 / frec  # Sample time
    N_horizont = 30
    t_prediction = N_horizont / frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Parameters of the system
    g = 9.8
    m0 = 1.0
    I_xx = 0.02
    L = [g, m0, I_xx]

    # Vector Initial conditions
    x = np.zeros((6, t.shape[0]+1-N_prediction), dtype = np.double)

    # Initial Control values
    u_control = np.zeros((2, t.shape[0]-N_prediction), dtype = np.double)
    #x_fut = np.ndarray((6, N_prediction+1))
    x_fut = np.zeros((6, 1, N_prediction+1))

    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)

    # Inicializar xref
    xref = np.zeros((8, t.shape[0]), dtype=np.double)


    # Calcular posiciones parametrizadas en longitud de arco
    arc_lengths, pos_ref= calculate_positions_in_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t, t_max=t_final)
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)
    # Calcular las derivadas de las posiciones con respecto a la longitud de arco


    xref[0, :] = pos_ref[0, :]  
    xref[1, :] = pos_ref[1, :]  

    xref[3,:] = dp_ds [0, :]     
    xref[4,:] = dp_ds [1, :]    

    # Inicializar el array para almacenar v_theta
    v_theta = np.zeros(len(t))

    # Load the model of the system
    model, f = f_system_model()

    # Maximiun Values
    f_max = 3*m0*g
    f_min = 0

    t_max = 0.3
    t_min = -t_max 

    bounded = [f_max, f_min, t_max, t_min]

    # Optimization Solver
    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, bounded)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    
    
    # Crear una matriz de matrices para almacenar las coordenadas (x, y) de cada punto para cada instante k
    puntos = 5
    
    left_poses = np.empty((puntos+1, 2, t.shape[0]+1-N_prediction), dtype = np.double)
    rigth_poses = np.empty((puntos+1, 2, t.shape[0]+1-N_prediction), dtype = np.double)

    j = 0


    print("AQUI TA")
    for k in range(0, t.shape[0]-N_prediction):

                 
        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        for i in range(N_prediction):
            x_fut[:, 0, i] = acados_ocp_solver.get(i, "x")

        
        x_fut[:, 0, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        print("CACA")
        
        # update yref
        for j in range(N_prediction):

            yref = xref[:,k+j]

            parameters = np.hstack([yref, 0,0, 0,0, 0])
            acados_ocp_solver.set(j, "p", parameters)
        
        yref_N = xref[:,k+N_prediction]
        parameters_N = np.hstack([yref_N, 0,0, 0,0, 0])
        acados_ocp_solver.set(N_prediction, "p", parameters_N)

        # Get Computational Time
        tic = time.time()
        # solve ocp
        status = acados_ocp_solver.solve()

        toc = time.time()- tic

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        # System Evolution
        x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
        delta_t[:, k] = toc

        print(k)

    # Ejemplo de uso
    fig3 = animate_triangle_pista(x[:3, :], xref[:2, :], left_poses[:, : , :], rigth_poses[:, :, :], 'animation.mp4')   

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')

        
if __name__ == '__main__':
    main()