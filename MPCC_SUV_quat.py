from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin, sqrt
from casadi import  dot
from casadi import nlpsol
from casadi import sumsqr
from casadi import power
from casadi import diff
from fancy_plots import plot_pose, fancy_plots_2, fancy_plots_1, plot_vel_norm
from animacion_SUV import animate_SUV_pista
from graf import animate_triangle
from plot_states import plot_states
from scipy.integrate import quad
from scipy.optimize import bisect
from casadi import dot, norm_2, mtimes, DM, SX, MX,  if_else
from casadi import atan2, tanh
from casadi import jacobian
from casadi import substitute
import os
from scipy.io import savemat


from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
#from graf import animate_triangle



# --- Obstáculo fijo (obs1) ---
obs1_x = 2.5     # [m] cruce lateral izquierdo
obs1_y = 4      # [m]
obs1_z = 0.0       # [m]
obs1_r = 0.6       # [m]

# --- Obstáculo móvil (obs2) ---
obs2_x = 2.5       # [m] lazo inferior derecho
obs2_y = -5      # [m]
obs2_z = 0.0       # [m]
obsmovil_r = 0.5   # [m]

# Radio del UAV
uav_r = 0.3        # [m] radio aproximado del dron
margen = 0.2       # [m] margen de seguridad

def QuatToRot(quat: MX):
    # quat: MX(4x1)
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    # Normalizar
    q_norm = sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0 /= q_norm
    q1 /= q_norm
    q2 /= q_norm
    q3 /= q_norm

    # Matriz de rotación 3x3
    Rot = MX.zeros(3, 3)
    Rot[0, 0] = 1 - 2*(q2**2 + q3**2)
    Rot[0, 1] = 2*(q1*q2 - q0*q3)
    Rot[0, 2] = 2*(q1*q3 + q0*q2)

    Rot[1, 0] = 2*(q1*q2 + q0*q3)
    Rot[1, 1] = 1 - 2*(q1**2 + q3**2)
    Rot[1, 2] = 2*(q2*q3 - q0*q1)

    Rot[2, 0] = 2*(q1*q3 - q0*q2)
    Rot[2, 1] = 2*(q2*q3 + q0*q1)
    Rot[2, 2] = 1 - 2*(q1**2 + q2**2)

    return Rot

def QuatToRot_numpy(quat: np.ndarray):
    q0, q1, q2, q3 = quat
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0, q1, q2, q3 = q0/norm, q1/norm, q2/norm, q3/norm

    Rot = np.zeros((3,3))
    Rot[0,0] = 1 - 2*(q2**2 + q3**2)
    Rot[0,1] = 2*(q1*q2 - q0*q3)
    Rot[0,2] = 2*(q1*q3 + q0*q2)

    Rot[1,0] = 2*(q1*q2 + q0*q3)
    Rot[1,1] = 1 - 2*(q1**2 + q3**2)
    Rot[1,2] = 2*(q2*q3 - q0*q1)

    Rot[2,0] = 2*(q1*q3 - q0*q2)
    Rot[2,1] = 2*(q2*q3 + q0*q1)
    Rot[2,2] = 1 - 2*(q1**2 + q2**2)
    return Rot


def smooth_quaternions(quat_seq):
    """
    Ajusta la secuencia de cuaterniones para evitar saltos
    de signo entre q y -q (misma orientación).
    quat_seq: np.array shape (4, N)
    """
    q_fixed = quat_seq.copy()
    for k in range(1, quat_seq.shape[1]):
        if np.dot(q_fixed[:,k-1], q_fixed[:,k]) < 0:
            q_fixed[:,k] *= -1
    return q_fixed


def f_system_model():

    model_name = 'USV_quat_ode'

    # --- Parámetros físicos tabla de surf robotizada (~10 kg) ---
    m = 10.0         # masa (kg)
    L = 1.6          # longitud típica de tabla (m)
    Iz = m * L**2 / 12   # momento de inercia aprox. en yaw
    # -> ≈ 2.1 kg·m²

    # Added mass (muy pequeños en tablas planas)
    Xu_dot = -0.5    # surge added mass
    Yv_dot = -3.0    # sway added mass
    Nr_dot = -0.5    # yaw added mass

    # Coeficientes de damping
    du = 0.3         # surge damping (fricción hidrodinámica baja)
    dv = 1.5         # sway damping (resistencia lateral)
    dr = 0.2         # yaw damping (muy chico porque no tiene quilla grande)


    # --- Estados ---
    x_pos = MX.sym('x')
    y_pos = MX.sym('y')
    q0 = MX.sym('q0')
    q1 = MX.sym('q1')
    q2 = MX.sym('q2')
    q3 = MX.sym('q3')
    u   = MX.sym('u')   # surge
    v   = MX.sym('v')   # sway
    r   = MX.sym('r')   # yaw rate
    x = vertcat(x_pos, y_pos, q0, q1, q2, q3, u, v, r)

    # --- Derivadas ---
    x_p  = MX.sym('x_p')
    y_p  = MX.sym('y_p')
    q0_p = MX.sym('q0_p')
    q1_p = MX.sym('q1_p')
    q2_p = MX.sym('q2_p')
    q3_p = MX.sym('q3_p')
    u_p  = MX.sym('u_p')
    v_p  = MX.sym('v_p')
    r_p  = MX.sym('r_p')
    xdot = vertcat(x_p, y_p, q0_p, q1_p, q2_p, q3_p, u_p, v_p, r_p)


    # --- Controles ---
    tau_u = MX.sym('tau_u')   # surge force
    tau_r = MX.sym('tau_r')   # yaw torque
    u_control = vertcat(tau_u, tau_r)


    # --- Parámetros externos ---
    # Estados deseados
    x_d   = MX.sym('x_d')
    y_d   = MX.sym('y_d')
    q0_d  = MX.sym('q0_d')
    q1_d  = MX.sym('q1_d')
    q2_d  = MX.sym('q2_d')
    q3_d  = MX.sym('q3_d')
    u_d   = MX.sym('u_d')
    v_d   = MX.sym('v_d')
    r_d   = MX.sym('r_d')

    # Controles deseados
    tau_u_d = MX.sym('tau_u_d')
    tau_r_d = MX.sym('tau_r_d')

    # Errores obligatorios
    el_x   = MX.sym('el_x')
    el_y   = MX.sym('el_y')
    ec_x   = MX.sym('ec_x')
    ec_y   = MX.sym('ec_y')
    theta_p = MX.sym('theta_p')

    # Vector de parámetros externos
    p = vertcat(
        x_d, y_d,
        q0_d, q1_d, q2_d, q3_d,
        u_d, v_d, r_d,
        tau_u_d, tau_r_d,
        el_x, el_y, ec_x, ec_y, theta_p
    )

    # --- Quaternion evolution ---
    quat = vertcat(q0, q1, q2, q3)
    S = vertcat(
        horzcat(0, 0, 0, -r),
        horzcat(0, 0, r,  0),
        horzcat(0, -r, 0, 0),
        horzcat(r, 0, 0, 0)
    )
    quat_dot = 0.5 * (S @ quat)

    # --- Cinemática rotacional---
    # --- Cinemática rotacional ---
    nu = vertcat(u, v, 0)   # velocidad en cuerpo
    J = QuatToRot(quat)     # matriz de rotación 3x3
    vel = J @ nu            # velocidad en mundo

    dx = vel[0]
    dy = vel[1]
    


    # --- Dinámica ---
    du_dt = (1/(m - Xu_dot)) * (tau_u - du*u + m*v*r)
    dv_dt = (1/(m - Yv_dot)) * (- dv*v - m*u*r)   # sin tau_v
    dr_dt = (1/(Iz - Nr_dot)) * (tau_r - dr*r)


    # Vector final de dinámica explícita
    f_expl = vertcat(dx, dy, quat_dot, du_dt, dv_dt, dr_dt)
    f_impl = xdot - f_expl

    # --- f_system para simulación ---
    f_system = Function('system', [x, u_control], [f_expl])
    
    # Define f_x and g_x
    # Parte libre: evalúa f(x,u) en u=0
    u_zero = MX.zeros(u_control.shape[0], 1)
    f0_expr = substitute(f_expl, u_control, u_zero)   # f0(x)

    # Crear funciones CasADi
    f_x = Function('f0', [x], [f0_expr])                  # dinámica libre
    g_x = Function('g', [x], [jacobian(f_expl, u_control)])  # incidencia de u

    # --- Modelo Acados ---
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = u_control
    model.p = p
    model.name = model_name

    return model, f_system, f_x, g_x



def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x_next = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    return np.squeeze(x_next)

def quaternionMultiply(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    scalarPart = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    vectorPart = vertcat(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    
    q_result = vertcat(scalarPart, vectorPart)
    return q_result

def quaternion_error(q_real, quat_d):
    norm_q = norm_2(q_real)
    q_inv = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    
    q_error = quaternionMultiply(q_inv, quat_d)

    return q_error

def log_cuaternion_casadi(q):
 
    # Descomponer el cuaternio en su parte escalar y vectorial
    q_w = q[0]
    q_v = q[1:]

    q = if_else(
        q_w < 0,
        -q,  # Si q_w es negativo, sustituir q por -q
        q    # Si q_w es positivo o cero, dejar q sin cambios
    )

    # Actualizar q_w y q_v después de cambiar q si es necesario
    q_w = q[0]
    q_v = q[1:]
    
    # Calcular la norma de la parte vectorial usando CasADi
    norm_q_v = norm_2(q_v)

    #print(norm_q_v)
    
    # Calcular el ángulo theta
    theta = atan2(norm_q_v, q_w)
    
    log_q = 2 * q_v * theta / norm_q_v
    
    return log_q

def create_ocp_solver_description(x0, N_horizon, t_horizon, bounded) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    F1_max = bounded[0] 
    F1_min = bounded[1] 
    F2_max = bounded[2]
    F2_min = bounded[3]
    F3_max = bounded[4]
    F3_min = bounded[5]

    
    model, f_system, f_x, g_x = f_system_model()
    ocp.model = model
    ocp.p = model.p
    
    
    # Calcula las dimensiones
    nx = model.x.shape[0]
    nu = model.u.shape[0]
    nparametros = 5

    ny = nx + nu + nparametros

    # set dimensions
    ocp.dims.N = N_horizon
    ocp.parameter_values = np.zeros(ny)

    # set cost
    Q_mat = 1.5 * np.diag([1, 1])  # [x,th,dx,dth]
    R_mat = 0.00001 * np.diag([1,  1])

    # Define matrices de ganancia para los errores
    Q_el = 5 * np.eye(2)  # Ganancia para el error el (2x2)
    Q_ec = 6 * np.eye(2)  # Ganancia para el error ec (2x2)
    Q_theta_p = 500  # Ganancia para theta_p (escalar)
    R_u = 1 * np.diag([0.005, 0.005])   # solo surge & yaw torque
    V_mat = 0*0.001* np.eye(2)  # Ganancia para el error ec (2x2)
    Q_vels = 0.001
    # Penalización de actitud: yaw >> roll,pitch
    Q_q = np.diag([
        0.1,   # roll (muy poco, casi libre)
        0.1,   # pitch (muy poco, casi libre)
        2   # yaw (prioridad fuerte)
    ])

    # Definir los errores como vectores
 

    # Definir variables simbólicas  
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ## ERROR DE ACtitud
    # Quaternion real (del estado)
    q_real = vertcat(model.x[2], model.x[3], model.x[4], model.x[5])

    # Quaternion deseado (de parámetros)
    q_des = vertcat(ocp.p[11], ocp.p[12], ocp.p[13], ocp.p[14])

    # Error de cuaternión
    q_error = quaternion_error(q_real, q_des)

    # Logaritmo de cuaternión (para penalizar en el costo)
    log_q = log_cuaternion_casadi(q_error)

    #ERROR DE POSICION
    sd = ocp.p[0:2]
    error_pose = sd - model.x[0:2]                        

    #ERROR DE ARRASTRE
    sd_p = ocp.p[3:5]
    tangent_normalized = sd_p / norm_2(sd_p)## ---> por propiedad la nomra de la recta tangente en longotud de arco ya es unitario
    el = dot(tangent_normalized, error_pose) * tangent_normalized

    # ERROR DE CONTORNO
    I = MX.eye(2) 
    P_ec = I - tangent_normalized.T @ tangent_normalized
    ec = P_ec @ error_pose 

    ## ERROR DE VELOCIDAD
    # Velocidad en cuerpo (surge, sway, 0)
    nu_body = vertcat(model.x[6], model.x[7], 0)

    # Matriz de rotación desde cuaternión
    quat = model.x[2:6]
    J = QuatToRot(quat)

    # Velocidad en mundo
    vel_inertial = J @ nu_body
    vel_inertial_xy = vel_inertial[0:2]

    # Proyección sobre la tangente de la referencia
    vel_progres = dot(tangent_normalized, vel_inertial_xy)



    # Define el costo externo considerando los errores como vectores
    actitud_cost = log_q.T @ Q_q @ log_q 
    error_contorno = ec.T @ Q_ec @ ec
    error_lag = el.T @ Q_el @ el

    vel_progres_cost = Q_vels*vel_progres  
    
    ocp.model.cost_expr_ext_cost = (actitud_cost) + (error_contorno + error_lag) - vel_progres_cost + 1*model.u.T @ R_u @ model.u 
    ocp.model.cost_expr_ext_cost_e = (actitud_cost) + (error_contorno + error_lag) - 1* vel_progres_cost

    # PRIMERA BARRERA (en el espacio 3D)




    # --- Funciones de barrera ---
    h = sqrt((model.x[0] - obs1_x)**2 +
            (model.x[1] - obs1_y)**2 +
            (model.x[2] - obs1_z)**2) - (uav_r + obs1_r + margen)

    h_movil = sqrt((model.x[0] - obs2_x)**2 +
                (model.x[1] - obs2_y)**2 +
                (model.x[2] - obs2_z)**2) - (uav_r + obsmovil_r + margen)


    f_x_val = f_x(model.x)
    g_x_val = g_x(model.x)   # solo el estado


    # Derivada de Lie de primer orden
    Lf_h = jacobian(h, model.x) @ f_x_val 
    Lg_h = jacobian(h, model.x) @ g_x_val

    Lf_h_movil = jacobian(h_movil, model.x) @ f_x_val 
    Lg_h_movil = jacobian(h_movil, model.x) @ g_x_val

    # Derivada de Lie de segundo orden
    Lf2_h = jacobian(Lf_h, model.x) @ f_x_val
    Lg_L_f_h = jacobian(Lf_h, model.x) @ g_x_val 
    
    Lf2_h_movil = jacobian(Lf_h_movil, model.x) @ f_x_val
    Lg_L_f_h_movil = jacobian(Lf_h_movil, model.x) @ g_x_val 

    # Barreras temporales
    h_p = Lf_h + Lg_h @ model.u
    h_pp = Lf2_h + Lg_L_f_h @ model.u

    h_p_movil = Lf_h_movil + Lg_h_movil @ model.u
    h_pp_movil = Lf2_h_movil + Lg_L_f_h_movil @ model.u

    # # set constraints
    ocp.constraints.constr_type = 'BGH'

    # Funciones de barrera de segundo orden
    nb_1 = h
    nb_2 = vertcat(h, Lf_h) 

    nb_1_movil = h_movil
    nb_2_movil = vertcat(h_movil, Lf_h_movil) 


    K_alpha = MX([25, 15]).T ## 20 8
    K_alpha_movil = MX([25, 15]).T ## 20 8

    #constraints = vertcat(h_p + 5*nb_1)
    CBF_static = h_pp +  K_alpha @ nb_2
    CBF_movil = h_pp_movil +  K_alpha_movil @ nb_2_movil

    constraints = vertcat(CBF_static, CBF_movil)
    #constraints = vertcat(CBF_static, CBF_movil  , V_p + 0.9*V, vel_progres)
    #constraints = vertcat(model.x[0] )

    # Asigna las restricciones al modelo del OCP
    N_constraints = constraints.size1()

    ocp.model.con_h_expr = constraints
    ocp.constraints.lh = np.array([0,   0 ])  # Límite inferior 
    ocp.constraints.uh = np.array([1e9, 1e9])  # Límite superior

    # Configuración de las restricciones suaves
    cost_weights =  np.array([1,1])
    ocp.cost.zu = 1*cost_weights 
    ocp.cost.zl = 1*cost_weights 
    ocp.cost.Zl = 1 * cost_weights 
    ocp.cost.Zu = 1 * cost_weights 

    # Índices para las restricciones suaves (necesario si se usan)
    ocp.constraints.idxsh = np.arange(N_constraints)  # Índices de las restricciones suaves
        
    ocp.constraints.x0 = x0

    F1_max = 300     # N
    F1_min = 0       # prohibido retroceso
    F2_max = 100     # Nm
    F2_min = -100    # Nm

    ocp.constraints.lbu = np.array([F1_min, F2_min])
    ocp.constraints.ubu = np.array([F1_max, F2_max])
    ocp.constraints.idxbu = np.array([0, 1])


    # set constraints
    # ocp.constraints.lbu = np.array([F1_min, F2_min, F3_min])
    # ocp.constraints.ubu = np.array([F1_max, F2_max, F3_max])
    # ocp.constraints.idxbu = np.array([0, 1, 2])

    #vmin = -8
    #vmax = 8
    #ocp.constraints.lbx = np.array([vmin,vmin])
    #ocp.constraints.ubx = np.array([vmax,vmax])
    #ocp.constraints.idxbx = np.array([3,4])

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.tol = 1e-4

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
value = 10

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

def calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):

    
    def r(t):
        """ Devuelve el punto en la trayectoria para el parámetro t usando las funciones de trayectoria. """
        return np.array([xd(t), yd(t), zd(t)])

    def r_prime(t):
        """ Devuelve la derivada de la trayectoria en el parámetro t usando las derivadas de las funciones de trayectoria. """
        return np.array([xd_p(t), yd_p(t), zd_p(t)])

    def integrand(t):
        """ Devuelve la norma de la derivada de la trayectoria en el parámetro t. """
        return np.linalg.norm(r_prime(t))

    def arc_length(tk, t0=0):
        """ Calcula la longitud de arco desde t0 hasta tk usando las derivadas de la trayectoria. """
        length, _ = quad(integrand, t0, tk)
        return length

    def find_t_for_length(theta, t0=0):
        """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
        func = lambda t: arc_length(t, t0) - theta
        return bisect(func, t0, t_max)

    # Generar las posiciones y longitudes de arco
    positions = []
    arc_lengths = []
    
    for tk in t_range:
        theta = arc_length(tk)
        arc_lengths.append(theta)
        point = r(tk)
        positions.append(point)

    arc_lengths = np.array(arc_lengths)
    positions = np.array(positions).T  # Convertir a array 2D (3, N)

    # Crear splines cúbicos para la longitud de arco con respecto al tiempo
    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])

    # Función que retorna la posición dado un valor de longitud de arco
    def position_by_arc_length(s):
        t_estimated = spline_t(s)  # Usar spline para obtener la estimación precisa de t
        return np.array([spline_x(t_estimated), spline_y(t_estimated), spline_z(t_estimated)])

    return arc_lengths, positions, position_by_arc_length

def calculate_reference_positions_and_curvature(arc_lengths,position_by_arc_length, t, t_s, v_max, alpha):
    # Calcular los valores de s para la referencia
    s_values = np.linspace(arc_lengths[0], arc_lengths[-1], len(arc_lengths))

    # Calcular las posiciones y sus derivadas con respecto a s
    positions = np.array([position_by_arc_length(s) for s in s_values])
    dr_ds = np.gradient(positions, s_values, axis=0)
    d2r_ds2 = np.gradient(dr_ds, s_values, axis=0)

    # Calcular la curvatura en cada punto
    cross_product = np.cross(dr_ds[:-1], d2r_ds2[:-1])
    numerator = np.linalg.norm(cross_product, axis=1)
    denominator = np.linalg.norm(dr_ds[:-1], axis=1)**3
    curvature = numerator / denominator

    # Definir la velocidad de referencia en función de la curvatura
    v_ref = v_max / (1 + alpha * curvature)

    # Inicializar s_progress y calcular el progreso en longitud de arco
    s_progress = np.zeros(len(t))
    s_progress[0] = s_values[0]
    for i in range(1, len(t)):
        s_progress[i] = s_progress[i-1] + v_ref[min(i-1, len(v_ref)-1)] * t_s

    # Calcular las posiciones de referencia basadas en el progreso de s
    pos_ref = np.array([position_by_arc_length(s) for s in s_progress])
    pos_ref = pos_ref.T

    # Calcular la derivada de la posición respecto a la longitud de arco
    dp_ds = np.gradient(pos_ref, s_progress, axis=1)

    return pos_ref, s_progress, v_ref, dp_ds

def calculate_orthogonal_error(error_total, tangent):

    if np.linalg.norm(tangent) == 0:
        return error_total  # No hay tangente válida, devolver el error total
    # Matriz de proyección ortogonal
    I = np.eye(2)  # Matriz identidad en 3D
    P_ec = I - np.outer(tangent, tangent)
    # Aplicar la matriz de proyección para obtener el error ortogonal
    e_c = P_ec @ error_total
    return e_c

def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]


def main():

    plt.figure(figsize=(10, 5))
    # Initial Values System
    #t_final = 7.58
    t_final = np.load("tref_optimo.npy")
    print(t_final)  # Ej: [12.34]
    frec = 30
    t_s = 1 / frec  # Sample time
    N_horizont = 30
    t_prediction = N_horizont / frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction, t_s)
    N_prediction = N.shape[0]

    # Time simulation
        ######################################
    # Cargar el archivo guardado
    xd_all = np.load("xref_optimo.npy")

    # Extraer cada señal
    xd_array = xd_all[0, :]
    yd_array = xd_all[1, :]
    zd_array = np.zeros_like(xd_array)  # Si es 2D, lo dejas en cero

    xd_p_array = xd_all[3, :]
    yd_p_array = xd_all[4, :]
    zd_p_array = np.zeros_like(xd_array)

    # Vector de tiempo correspondiente
    t = np.linspace(0, t_final, xd_array.shape[0])

    # Crear funciones de interpolación
    xd = interp1d(t, xd_array, kind='cubic')
    yd = interp1d(t, yd_array, kind='cubic')
    zd = interp1d(t, zd_array, kind='cubic')

    xd_p = interp1d(t, xd_p_array, kind='cubic')
    yd_p = interp1d(t, yd_p_array, kind='cubic')
    zd_p = interp1d(t, zd_p_array, kind='cubic')


    ###########################################
    #t = np.arange(0, t_final, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    #Vectores
    h_CBF_1 = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    h_CBF_2 = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    CLF = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    e_contorno = np.zeros((2, t.shape[0] - N_prediction), dtype=np.double)
    e_arrastre = np.zeros((2, t.shape[0] - N_prediction), dtype=np.double)
    e_total = np.zeros((2, t.shape[0] - N_prediction), dtype=np.double)
    vel_progres = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    vel_progress_ref =  np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)

    # Parameters of the system
    g = 9.8
    m0 = 1.0
    I_xx = 0.02
    L = [g, m0, I_xx]

    # Vector Initial conditions
    x = np.zeros((9, t.shape[0]+1-N_prediction), dtype = np.double)

    # Initial Control values
    u_control = np.zeros((2, t.shape[0]-N_prediction), dtype = np.double)
    #x_fut = np.ndarray((6, N_prediction+1))
    x_fut = np.zeros((9, 1, N_prediction+1))

    #xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)


    # Calcular psid y su derivada
    xd_p_vals = xd_p(t)
    yd_p_vals = yd_p(t)
    psid = np.arctan2(yd_p_vals, xd_p_vals)


    #quaternion = euler_to_quaternion(0, 0, psid) 
    quatd= np.zeros((4, t.shape[0]), dtype = np.double)


    # Calcular los cuaterniones utilizando la función euler_to_quaternion para cada psid
    for i in range(t.shape[0]):
        quaternion = euler_to_quaternion(0, 0, psid[i])  # Calcula el cuaternión para el ángulo de cabeceo en el instante i
        quatd[:, i] = quaternion  # Almacena el cuaternión en la columna i de 'quatd'
        #quatd[:, i] = [1, 0, 0 ,0]
    
    # Inicializar xref
    xref = np.zeros((11, t.shape[0]), dtype=np.double)

    vel_norm = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    vel_ref_norm = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)

    # Calcular posiciones parametrizadas en longitud de arco
    #arc_lengths, pos_ref= calculate_positions_in_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t, t_max=t_final)
    arc_lengths, pos_ref, position_by_arc_length = calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t, t_max=t_final)
    
    print(t[0:10])
    print(arc_lengths[0:10])


    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)


    # VERIFICACION DE LONGITUD DE ARCO
    t_value = 0.3
    pos_by_t = np.array([xd(t_value), yd(t_value), zd(t_value)])


    # Calcular la longitud de arco correspondiente a t_value
    #arc_length_at_t = arc_lengths[np.argmin(np.abs(t - t_value))]  # Longitud de arco más cercana a t_value


    pos_by_arc_length = position_by_arc_length(2.78851822)


    # Imprimir ambos valores
    print("Posición calculada a partir de t:", pos_by_t)
    print("Posición calculada a partir de la longitud de arco:", pos_by_arc_length)

    #time.sleep(20)
        
    xref[0, :] = pos_ref[0, :]  
    xref[1, :] = pos_ref[1, :]  

    xref[3,:] = dp_ds [0, :]     
    xref[4,:] = dp_ds [1, :]    

    # Inicializar el array para almacenar v_theta
    v_theta = np.zeros(len(t))

    # Load the model of the system
    model, f, f_x, g_x = f_system_model()

    # Maximiun Values


    F1_max = 1
    F2_max = 1
    F3_max = 1

    F1_min = -F1_max
    F2_min = -F2_max
    F3_min = -F3_max

    bounded = [F1_max, F1_min, F2_max, F2_min, F3_max, F3_min]

    # Optimization Solver
    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, bounded)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # ESTADOS INICALES
    # ejemplo: barco en el origen, orientado al eje X (yaw = 0)
    x_pos0, y_pos0 = 0.0, 0.0
    psi0 = 0.0  

    q0 = 0.70710678
    q1 = 0.0
    q2 = 0.0
    q3 = 0.70710678


    u0, v0, r0 = 0.0, 0.0, 0.0

    x_init = np.array([x_pos0, y_pos0, q0, q1, q2, q3, u0, v0, r0])


    for i in range(3, 0, -1):
        print(f"Leyendo odometría... Inicio en {i} segundos")
        # Realizar exactamente 5 lecturas en 1 segundo
        for _ in range(3):
            x[:, 0] = x_init
            time.sleep(0.2)  # Espera 0.2s entre cada lectura (5 lecturas en 1 segundo)
    print("¡Sistema listo!") 


    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x_init)

    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    
    
    # Crear una matriz de matrices para almacenar las coordenadas (x, y) de cada punto para cada instante k
    puntos = 5
    
    left_poses = np.empty((puntos+1, 2, t.shape[0]+1-N_prediction), dtype = np.double)
    rigth_poses = np.empty((puntos+1, 2, t.shape[0]+1-N_prediction), dtype = np.double)

    j = 0

    obst_static = np.array([obs1_x, obs1_y])
    obst_movil  = np.array([obs2_x, obs2_y])


    print("Calculando...")
    
    for k in range(0, t.shape[0]-N_prediction):

        # Posición actual del vehículo en 2D
        pos_vehicle = np.array([x[0, k], x[1, k]])

        # Funciones de barrera h(x) en 2D
        h_CBF_1[:, k] = np.linalg.norm(pos_vehicle - obst_static) - (uav_r + obs1_r + margen)
        h_CBF_2[:, k] = np.linalg.norm(pos_vehicle - obst_movil)  - (uav_r + obsmovil_r + margen)

                 
        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        for i in range(N_prediction):
            x_fut[:, 0, i] = acados_ocp_solver.get(i, "x")
      
        x_fut[:, 0, N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        # update yref
        for j in range(N_prediction):

            yref = xref[:,k+j]
            parameters = np.hstack([yref, quatd[:,k+j], 0])
            acados_ocp_solver.set(j, "p", parameters)
        
        yref_N = xref[:,k+N_prediction]
        parameters_N = np.hstack([yref_N, quatd[:,k+N_prediction], 0])
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

        # ==============================
        # Cálculo de errores y velocidad de progreso
        # ==============================
        

        # Estado actual en 2D
        pos_vehicle = x[0:2, k]

        # Referencia en 2D
        pos_ref_k = xref[0:2, k]
        dp_ds_k   = xref[3:5, k]     # tangente a la trayectoria

        # Error total de posición
        error_pose = pos_ref_k - pos_vehicle

        # Vector tangente unitario
        tangent_norm = dp_ds_k / np.linalg.norm(dp_ds_k)

        # Error de arrastre (paralelo al tangente)
        e_arrastre[:, k] = (np.dot(tangent_norm, error_pose)) * tangent_norm

        # Error de contorno (perpendicular al tangente)
        I = np.eye(2)
        P_ec = I - np.outer(tangent_norm, tangent_norm)
        e_contorno[:, k] = P_ec @ error_pose

        # Error total
        e_total[:, k] = error_pose

        # Velocidad en mundo
        quat = x[2:6, k]   # estado actual (np.array)
        nu_body = np.array([x[6, k], x[7, k], 0.0])
        J = QuatToRot_numpy(quat)   # ahora sí en NumPy
        vel_inertial = J @ nu_body
        vel_norm[:, k] = np.linalg.norm(vel_inertial[0:2])

        # Proyección sobre el tangente
        vel_progres[:, k] = np.dot(tangent_norm, vel_inertial[0:2])


    # Ejemplo de uso
    
    print("Graficando...")  

    obstacles = [
        (obs1_x, obs1_y, obs1_r),     # obstáculo fijo
        (obs2_x, obs2_y, obsmovil_r)  # obstáculo móvil
    ]

    # Llamada igual que antes, solo con obstacles extra
    fig3 = animate_SUV_pista(
        x[0:6, :],
        xref[0:2, :],
        left_poses[:, :, :],
        rigth_poses[:, :, :],
        'animation_MPCC_SUV.mp4',
        obstacles=obstacles
    )



    # ============================
    # 5) Estados x,y: Real vs Deseado
    # ============================

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(t[:x.shape[1]], x[0,:], label='x real', color='blue')
    plt.plot(t[:xref.shape[1]], xref[0,:], '--', label='x deseado', color='blue', alpha=0.6)
    plt.ylabel("x [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t[:x.shape[1]], x[1,:], label='y real', color='red')
    plt.plot(t[:xref.shape[1]], xref[1,:], '--', label='y deseado', color='red', alpha=0.6)
    plt.ylabel("y [m]")
    plt.xlabel("Tiempo [s]")
    plt.grid(True)
    plt.legend()

    plt.suptitle("5) Estados: x, y - Real vs Deseado")
    plt.tight_layout()
    plt.savefig("5_xy_vs_ref.png")
    plt.close()
    # ============================
    # 1) Velocidades en cuerpo
    # ============================
# ============================
# 1) Velocidades en cuerpo
# ============================
    plt.figure(figsize=(10,6))

    plt.subplot(3,1,1)
    plt.plot(t[:x.shape[1]], x[6,:], label='u (surge)')
    plt.ylabel("u [m/s]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(t[:x.shape[1]], x[7,:], label='v (sway)', color='orange')
    plt.ylabel("v [m/s]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t[:x.shape[1]], x[8,:], label='r (yaw rate)', color='green')
    plt.ylabel("r [rad/s]")
    plt.xlabel("Tiempo [s]")
    plt.grid(True)
    plt.legend()

    plt.suptitle("1) Velocidades en el cuerpo (u, v, r)")
    plt.tight_layout()
    plt.savefig("1_velocidades.png")
    plt.close()



    # ============================
    # 2) Acciones de control
    # ============================
    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(t[:u_control.shape[1]], u_control[0,:], label='Fuerza Surge', color='blue')
    plt.ylabel("Surge [N]")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(t[:u_control.shape[1]], u_control[1,:], label='Torque Yaw', color='red')
    plt.ylabel("Yaw [Nm]")
    plt.xlabel("Tiempo [s]")
    plt.grid(True)
    plt.legend()

    plt.suptitle("2) Acciones de control")
    plt.tight_layout()
    plt.savefig("2_controles.png")
    plt.close()

    # ============================
    # 3) Trayectoria (posiciones) + Obstáculos
    # ============================

    plt.figure(figsize=(8,6))
    plt.plot(x[0,:], x[1,:], label="Posición real", color='blue')
    plt.plot(xref[0,:], xref[1,:], '--', label="Posición deseada", color='red')

    # Dibujar obstáculos como círculos
    circle1 = plt.Circle((obs1_x, obs1_y), obs1_r, color='green', alpha=0.3, label="Obstáculo fijo")
    circle2 = plt.Circle((obs2_x, obs2_y), obsmovil_r, color='purple', alpha=0.3, label="Obstáculo móvil")

    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("3) Trayectoria: Real vs Deseada + Obstáculos")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("3_trayectoria.png")
    plt.close()


    # ============================
    # 4) Cuaterniones REAL vs DESEADO
    # ============================

    #q_real = smooth_quaternions(x[2:6, :])       # suavizar saltos
    #q_des  = smooth_quaternions(quatd[:, :q_real.shape[1]])

    q_real = (x[2:6, :])       # suavizar saltos
    q_des  = quatd[:, :q_real.shape[1]]

    plt.figure(figsize=(12,8))
    labels = ['q0', 'q1', 'q2', 'q3']

    for i in range(4):
        plt.subplot(4,1,i+1)
        plt.plot(t[:q_real.shape[1]], q_real[i,:], label=f'{labels[i]} real')
        plt.plot(t[:q_des.shape[1]], q_des[i,:], '--', label=f'{labels[i]} deseado')
        plt.ylabel(labels[i])
        plt.grid(True)
        plt.legend()

    plt.xlabel("Tiempo [s]")
    plt.suptitle("4) Cuaterniones: Real vs Deseado (sin saltos)")
    plt.tight_layout()
    plt.savefig("4_quat_comparacion.png")
    plt.close()

        # ============================
    # 6) Funciones de barrera CBF
    # ============================
    plt.figure(figsize=(10,6))
    plt.plot(t[:h_CBF_1.shape[1]], h_CBF_1[0,:], label="CBF estático", color='blue')
    plt.plot(t[:h_CBF_2.shape[1]], h_CBF_2[0,:], label="CBF móvil", color='red')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("h(x)")
    plt.title("6) Funciones de barrera CBF")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("6_CBF.png")
    plt.close()

    # ============================
    # 7) Errores de seguimiento
    # ============================
    plt.figure(figsize=(10,8))

    plt.subplot(3,1,1)
    plt.plot(t[:e_total.shape[1]], e_total[0,:], label="Error total x", color='blue')
    plt.plot(t[:e_total.shape[1]], e_total[1,:], label="Error total y", color='red')
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.legend()
    plt.title("7) Errores de seguimiento")

    plt.subplot(3,1,2)
    plt.plot(t[:e_arrastre.shape[1]], e_arrastre[0,:], label="Error arrastre x", color='blue')
    plt.plot(t[:e_arrastre.shape[1]], e_arrastre[1,:], label="Error arrastre y", color='red')
    plt.ylabel("Error arrastre [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(t[:e_contorno.shape[1]], e_contorno[0,:], label="Error contorno x", color='blue')
    plt.plot(t[:e_contorno.shape[1]], e_contorno[1,:], label="Error contorno y", color='red')
    plt.ylabel("Error contorno [m]")
    plt.xlabel("Tiempo [s]")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("7_errores.png")
    plt.close()



        # ============================
    # 8) Velocidades: progreso vs norma real y ref
    # ============================
    plt.figure(figsize=(10,6))
    plt.plot(t[:vel_progres.shape[1]], vel_progres[0,:], label="Velocidad de progreso", color='green')
    plt.plot(t[:vel_norm.shape[1]], vel_norm[0,:], '--', label="Velocidad real (norma)", color='blue')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Velocidad [m/s]")
    plt.title("8) Velocidades: progreso, real")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("8_velocidades.png")
    plt.close()


    # --- Guardar resultados para MATLAB ---
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/MPCC_USV"

    # Verificar si la ruta no existe
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Estableciendo la ruta local como pwd.")
        pwd = os.getcwd()  # ruta local

    # Nombre del experimento
    experiment_number = 1
    name_file = "Results_MPCC_USV" + str(experiment_number) + ".mat"

    save = True
    if save:
        savemat(os.path.join(pwd, name_file), {
            'time': t,
            'states': x,
            'q_real': q_real,
            'q_d': q_des,
            'ref': xref,
            'controls': u_control,
            'CBF_1': h_CBF_1,
            'CBF_2': h_CBF_2,
            'e_total': e_total,
            'e_contorno': e_contorno,
            'e_arrastre': e_arrastre,
            'vel_progres': vel_progres,
            'vel_norm': vel_norm,
            'vel_ref_norm': vel_ref_norm,
            'obstacles': np.array([
                [obs1_x, obs1_y, obs1_r],
                [obs2_x, obs2_y, obsmovil_r]
            ])
        })



   


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')

        
if __name__ == '__main__':
    main()