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
from casadi import sin, sqrt
from casadi import  dot
from casadi import nlpsol
from casadi import sumsqr
from casadi import power
from casadi import diff
from fancy_plots import plot_pose, fancy_plots_2, fancy_plots_1, plot_vel_norm
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
    Q_el = 5 * np.eye(2)  # Ganancia para el error el (2x2)
    Q_ec = 5 * np.eye(2)  # Ganancia para el error ec (2x2)
    Q_theta_p = 500  # Ganancia para theta_p (escalar)
    R_u = 1 * np.diag([0.01, 0.005])
    V_mat = 0*0.001* np.eye(2)  # Ganancia para el error ec (2x2)
    Q_vels = 0.1

    # Definir los errores como vectores
 

    # Definir variables simbólicas  
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    
    #ERROR DE POSICION
    sd = ocp.p[0:2]
    error_pose = sd - model.x[0:2]                        

    error_actitud = 0 - model.x[2]

    #ERROR DE ARRASTRE
    sd_p = ocp.p[3:5]
    tangent_normalized = sd_p # sd_p / norm_2(sd_p) ---> por propiedad la nomra de la recta tangente en longotud de arco ya es unitario
    el = dot(tangent_normalized, error_pose) * tangent_normalized

    # ERROR DE CONTORNO
    I = MX.eye(2) 
    P_ec = I - tangent_normalized.T @ tangent_normalized
    ec = P_ec @ error_pose 

    ## ERROR DE VELOCIDAD
    vel_progres = dot(tangent_normalized, model.x[3:5])

    # Define el costo externo considerando los errores como vectores
    error_cost = 3*error_actitud**2 
    error_contorno = ec.T @ Q_ec @ ec
    error_lag = el.T @ Q_el @ el

    vel_progres_cost = Q_vels*vel_progres  
    
    ocp.model.cost_expr_ext_cost = (error_cost) + (error_contorno + error_lag) - vel_progres_cost + 1*model.u.T @ R_u @ model.u 
    ocp.model.cost_expr_ext_cost_e = (error_cost) + (error_contorno + error_lag) - 1* vel_progres_cost

    # set constraints
    ocp.constraints.constr_type = 'BGH'
    
    vx = model.x[3]
    vy = model.x[4]

    restriccion_norm = sqrt(vx**2 + vy**2) 
    constraints = vertcat(vx,vy)
    ocp.model.con_h_expr = constraints
    Dim_constraints = 2  # Define el número total de restricciones

    ocp.constraints.lh = np.array([-15,-15])  # Límite inferior 
    ocp.constraints.uh = np.array([15,15])  # Límite superior

    cost_weights = 1*np.ones(Dim_constraints)  
    ocp.cost.zu = 1 * cost_weights  
    ocp.cost.zl = 1 * cost_weights  
    ocp.cost.Zl = 10 * cost_weights  
    ocp.cost.Zu = 10 * cost_weights
    # Índices para las restricciones suaves (necesario si se usan)
    ocp.constraints.idxsh = np.arange(Dim_constraints)  # Índices de las restricciones suaves
        
    ocp.constraints.x0 = x0

    # set constraints
    ocp.constraints.lbu = np.array([F_min, T_min])
    ocp.constraints.ubu = np.array([F_max, T_max])
    ocp.constraints.idxbu = np.array([0, 1])

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

    #xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)



    

    # Inicializar xref
    xref = np.zeros((8, t.shape[0]), dtype=np.double)

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
    model, f = f_system_model()

    # Maximiun Values
    f_max = 9*m0*g
    f_min = -f_max 

    t_max = 0.2
    t_min = -t_max 

    bounded = [f_max, f_min, t_max, t_min]

    # Optimization Solver
    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, bounded)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
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

        #vel_norm[k] = np.linalg.norm(x[3:5,k])
        #print("Shape of x:", x.shape)
        vel_norm[:, k]= np.linalg.norm(x[3:5, k])

        

        print(vel_norm[:, k] )

    # Ejemplo de uso
    
    fig3 = animate_triangle_pista(x[:3, :], xref[:2, :], left_poses[:, : , :], rigth_poses[:, :, :], 'animation_MPCC.mp4')   

    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")
    plt.close(fig1)  # Cierra la figura después de guardarla

   


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')

        
if __name__ == '__main__':
    main()