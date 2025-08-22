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

from matplotlib.animation import FuncAnimation
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
    


    
    p = vertcat(nx_d, ny_d, psi_d, nx_p_d, ny_p_d, psi_p_d, F_d , T_d , xl_1 , yl_1, xl_2 , yl_2)

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
    variables_adicionales = 4
    ny = nx + nu + variables_adicionales

    # set dimensions
    ocp.dims.N = N_horizon
    ocp.parameter_values = np.zeros(ny)

    # set cost
    Q_mat = 20* np.diag([1, 1, 0.2, 0.0, 0.0, 0.0])  # [x,th,dx,dth]
    R_mat = 0.01 * np.diag([1,  1])

    ocp.cost.cost_type = "EXTERNAL"
    #ocp.cost.cost_type_e = "EXTERNAL"
    
    error_pose = ocp.p[0:6] - model.x[0:6]
    ocp.model.cost_expr_ext_cost = 0*error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u 
    ocp.model.cost_expr_ext_cost_e = 0*error_pose.T @ Q_mat @ error_pose

    # Definir el tipo de restricciones
    ocp.constraints.constr_type = 'BGH'

    x = model.x
    u = model.u

    # Define symbolic variables for the left and right bounds based on parameters
    s_left_x = ocp.p[8]  # Assuming the left boundary x-coordinate is at parameter index 
    s_left_y = ocp.p[9]  # Assuming the left boundary y-coordinate is at parameter index 
    s_right_x = ocp.p[10]  # Assuming the right boundary x-coordinate is at parameter index 
    s_right_y = ocp.p[11]  # Assuming the right boundary y-coordinate is at parameter index 

    s_left = vertcat(s_left_x, s_left_y)
    s_right = vertcat(s_right_x, s_right_y)

    # Constraints to ensure the UAV stays outside the region between the boundaries
    lower_bound_constraint_x = s_left[0] - x[0]
    lower_bound_constraint_y = s_left[1] - x[1]
    upper_bound_constraint_x = x[0] - s_right[0]
    upper_bound_constraint_y = x[1] - s_right[1]


    # Concatenate constraints
    constraints = vertcat(lower_bound_constraint_x, lower_bound_constraint_y, upper_bound_constraint_x, upper_bound_constraint_y)


    # Set the expression for the constraints in the model
    ocp.model.con_h_expr = constraints
    Dim_constraints = 4 

    print("HERE2")
    ocp.constraints.lh = np.array([0,0,0,0])  # Límite inferior 
    ocp.constraints.uh = np.array([1e9,1e9,1e9,1e9])  # Límite superior

    # Configuración de las restricciones suaves
    cost_weights = np.ones(Dim_constraints)  # Pesos para las restricciones suaves
    ocp.cost.zu = 1*cost_weights  # Pesos para la penalización en el término de costo de las restricciones suaves (superior)
    ocp.cost.zl = 1*cost_weights  # Pesos para la penalización en el término de costo de las restricciones suaves (inferior)
    ocp.cost.Zl = 1000 * cost_weights  # Escala para el costo de las restricciones suaves (inferior)
    ocp.cost.Zu = 1000 * cost_weights  # Escala para el costo de las restricciones suaves (superior)

    # Índices para las restricciones suaves (necesario si se usan)
    ocp.constraints.idxsh = np.arange(Dim_constraints)  # Índices de las restricciones suaves

    
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




def main():

    plt.figure(figsize=(10, 5))
    # Initial Values System
    # Simulation Time
    t_final = 15
    # Sample time
    frecuencia = 30
    t_s = 1/frecuencia
    # Prediction Time
    #t_prediction= 2
    Horizont = 50

    t_prediction = Horizont/frecuencia

    # Nodes inside MPC
    #N = np.arange(0, t_prediction + t_s, t_s)
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0] #  devuelve la longitud o cantidad de elementos en N.
    print(N_prediction)

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Parameters of the system
    g = 9.8
    m0 = 0.5
    I_xx = 0.0002
    L = [g, m0, I_xx]


    # Vector Initial conditions
    x = np.zeros((6, t.shape[0]+1-N_prediction), dtype = np.double)

    x[1,0] = 5.0

    # Initial Control values
    u_control = np.zeros((2, t.shape[0]-N_prediction), dtype = np.double)

    # Reference Trajectoryz
    # Reference Signal of the system
    xref = np.zeros((8, t.shape[0]), dtype = np.double)
    # Lemniscata simple de Bernoulli (proyección en XY)
    A = 2.0    # amplitud en X
    B = 4    # amplitud en Y
    w = 0.5    # frecuencia angular

    xref[0, :] = A * np.sin(w * t)                  # X
    xref[1, :] = B * np.sin(w * t) * np.cos(w * t)  # Y  (desfase para formar el 8)

    # Mantener orientación constante
    xref[2, :] = 0.0
 
    # Load the model of the system
    model, f = f_system_model()

    # Maximiun Values
    f_max = 9*m0*g
    f_min = -f_max

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
    
    #GENERACION DE LA PISTA
    normal_x, normal_y = calculate_unit_normals(t, xref)
    track_width = 1

    Gl_funcion  = np.zeros((1, t.shape[0]+1-N_prediction), dtype = np.double)
    Gr_funcion  = np.zeros((1, t.shape[0]+1-N_prediction), dtype = np.double)
    
    # Parameters
    num_points_per_segment = 5  # Number of points to be used per segment
    spacing_between_points = 2 # Spacing between points

    # Create matrices to store the coordinates (x, y) of each point for each instant k
    left_trajectory = np.empty((num_points_per_segment + 1, 2, t.shape[0] + 1 - N_prediction), dtype=np.double)
    right_trajectory = np.empty((num_points_per_segment + 1, 2, t.shape[0] + 1 - N_prediction), dtype=np.double)

    # Create vectors to store the arc length and corresponding spline evaluations for each step
    spline_arc_length_left = np.zeros((2, t.shape[0] + 1 - N_prediction), dtype=np.double)
    spline_arc_length_right = np.zeros((2, t.shape[0] + 1 - N_prediction), dtype=np.double)

    # Initialize a matrix to store yref values
    
    # Initialize the matrix to store yref values
    yref_trajectory = np.zeros((xref.shape[0], N_prediction, t.shape[0] - N_prediction))


    j = 0

    for i in range(1, t.shape[0] - N_prediction):
        left_displacement = -0.5 * track_width
        right_displacement = 0.5 * track_width

        # Select points at certain intervals, ensuring no out-of-bounds index
        end_index = min(i + num_points_per_segment * spacing_between_points, xref.shape[1])
        indices = range(i - 1, end_index, spacing_between_points)

        left_x, left_y = displace_points_along_normal(
            xref[0, indices], xref[1, indices], normal_x[indices], normal_y[indices], left_displacement
        )
        right_x, right_y = displace_points_along_normal(
            xref[0, indices], xref[1, indices], normal_x[indices], normal_y[indices], right_displacement
        )

        # Calculate the cumulative arc length along the curve (s_k)
        left_arc_length = np.cumsum(np.sqrt(np.diff(left_x)**2 + np.diff(left_y)**2))
        left_arc_length = np.insert(left_arc_length, 0, 0)  # Insert 0 at the start
        right_arc_length = np.cumsum(np.sqrt(np.diff(right_x)**2 + np.diff(right_y)**2))
        right_arc_length = np.insert(right_arc_length, 0, 0)  # Insert 0 at the start

        # Fit cubic splines based on the arc length
        spline_left = CubicSpline(left_arc_length, np.vstack((left_x, left_y)).T, axis=0)
        spline_right = CubicSpline(right_arc_length, np.vstack((right_x, right_y)).T, axis=0)

        # Evaluate the splines along the original distances to obtain a smoothed trajectory (Evaluate the splines at s_k)
        spline_left_evaluated = spline_left(left_arc_length)
        spline_right_evaluated = spline_right(right_arc_length)

        # Store the spline trajectories in history matrices
        left_trajectory[:, :, j] = spline_left_evaluated
        right_trajectory[:, :, j] = spline_right_evaluated

        # Store the first points of the splines
        spline_arc_length_left[:, j] = spline_left_evaluated[0]
        spline_arc_length_right[:, j] = spline_right_evaluated[0]

        j += 1
        print(j)

    # Configure the figure and axes
    fig, ax = plt.subplots()
    val = 2
    ax.set_xlim(np.min(xref[0, :]) - val, np.max(xref[0, :]) + val)
    ax.set_ylim(np.min(xref[1, :]) - val, np.max(xref[1, :]) + val)

    # Plot the left and right splines and the evaluated points
    line_left, = ax.plot([], [], 'r-', lw=2, label='Left Spline')
    line_right, = ax.plot([], [], 'b-', lw=2, label='Right Spline')
    point_left, = ax.plot([], [], 'go', label="Evaluated Left Point")
    point_right, = ax.plot([], [], 'mo', label="Evaluated Right Point")

    ax.legend()
    # Función de actualización para la animación
    def update(frame):
        line_left.set_data(left_trajectory[:, 0, frame], left_trajectory[:, 1, frame])
        line_right.set_data(right_trajectory[:, 0, frame], right_trajectory[:, 1, frame])
        
        # Actualizar los primeros puntos evaluados en las splines
        point_left.set_data(spline_arc_length_left[0, frame], spline_arc_length_left[1, frame])
        point_right.set_data(spline_arc_length_right[0, frame], spline_arc_length_right[1, frame])
        
        return line_left, line_right, point_left, point_right

    # Crear la animación
    ani = FuncAnimation(fig, update, frames=range(j), blit=True, interval=100)

    # Guardar la animación en un archivo MP4
    ani.save('animation_splines.mp4', writer='ffmpeg', codec='h264', fps=10)

    # Mostrar la animación
    plt.show()    
   
    print("AQUI TA")
    for k in range(0,t.shape[0]-N_prediction):


        ## SECCION PARA GRAFICAR Y SACAR LOS PUNTOS FUTUROS
        left_pos = left_trajectory[:, :, k]
        rigth_pos = right_trajectory[:, :, k]

        Gl_funcion[0,k] = 1
        Gr_funcion[0,k] = 1
                  
        print(k)
        
        #COMIENZA EL PROGRAMA OPC

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # update yref
        for j in range(N_prediction):
            if k + j >= left_trajectory.shape[2]:
                break  # Exit the loop if the index is out of bounds
            yref = xref[:,k + j]
            left_bound = left_trajectory[:, :, k + j]
            right_bound = right_trajectory[:, :, k + j]

            # The parameters passed here should include yref, s_left, and s_right
            parameters = np.hstack([ yref, left_bound[0, 0], left_bound[0, 1], right_bound[0, 0], right_bound[0, 1]])
            acados_ocp_solver.set(j, "p", parameters)

            yref_trajectory[:,j, k] = yref  # Store yref in the matrix
        
        
        yref_N = xref[:, k + N_prediction]
        if k + N_prediction < left_trajectory.shape[2]:
            left_bound_N = left_trajectory[:, :, k + N_prediction]
            right_bound_N = right_trajectory[:, :, k + N_prediction]
            # Continue with processing
        else:
            # Handle the case where the index is out of bounds
            break
        parameters_N = np.hstack([ yref_N, left_bound_N[0, 0], left_bound_N[0, 1], right_bound_N[0, 0], right_bound_N[0, 1]])
        acados_ocp_solver.set(N_prediction, "p", parameters_N)
        
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


    
    # Ejemplo de uso
    #fig3 = animate_triangle_pista(x[:3, :], xref[:2, :], left_trajectory[:, : , :], right_trajectory[:, :, :], 'animation.mp4')   
    # Example of usage
    fig3 = animate_triangle_pista(x[:3, :], xref[:2, :], left_trajectory[:, : , :], right_trajectory[:, :, :], 'animation.mp4')

    #fig3 = animate_triangle(x[:3, :], xref[:2, :], 'animation.mp4')

   # plot_states(x[:3, :], xref[:3, :], 'states_plot.png')

     
    fig1 = plot_pose(x, xref, t)
    fig1.savefig("1_pose.png")


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')

        
if __name__ == '__main__':
    main()
