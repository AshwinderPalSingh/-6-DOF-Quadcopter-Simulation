import numpy as np
from scipy.spatial.transform import Rotation

class Quadcopter:
    """
    Quadcopter physics model (6-DOF, 12-state).
    
    State vector:
    x = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    
    Where:
    - (x, y, z): Position in world frame (m)
    - (vx, vy, vz): Velocity in world frame (m/s)
    - (phi, theta, psi): Euler angles (roll, pitch, yaw) in radians
    - (p, q, r): Angular velocities in body frame (rad/s)
    """
    def __init__(self, m=0.5, L=0.2, Ixx=0.002, Iyy=0.002, Izz=0.004, kf=1.0, km=0.01):
        # Parameters
        self.m = m         # Mass (kg)
        self.L = L         # Arm length (m)
        self.I = np.diag([Ixx, Iyy, Izz])  # Inertia tensor
        self.I_inv = np.linalg.inv(self.I)
        self.kf = kf       # Thrust coefficient
        self.km = km       # Torque coefficient
        self.g = 9.81      # Gravity (m/s^2)

        # Pre-allocate motor inputs from coefficients
        # [U1, U2, U3, U4] = A * [w1^2, w2^2, w3^2, w4^2]
        # U1 = Total Thrust, U2 = Roll Torque, U3 = Pitch Torque, U4 = Yaw Torque
        self.allocation_matrix = np.array([
            [self.kf, self.kf, self.kf, self.kf],
            [0, -self.L * self.kf, 0, self.L * self.kf],
            [self.L * self.kf, 0, -self.L * self.kf, 0],
            [-self.km, self.km, -self.km, self.km]
        ])

        # State vector [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.zeros(12)
        # Set initial z-position to be slightly off ground to avoid issues
        self.state[2] = 0.1 

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def _state_derivative(self, t, state):
        """
        Computes the derivative of the state vector.
        This is the core of the Equations of Motion (EOM).
        """
        # Unpack state
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        ang_vel = state[9:12]

        phi, theta, psi = angles
        p, q, r = ang_vel

        # 1. Get Rotation Matrix (Body to World)
        # Using ZYX convention (yaw, pitch, roll)
        R = Rotation.from_euler('zyx', [psi, theta, phi]).as_matrix()

        # 2. Get control inputs (Thrust and Torques)
        # These are passed externally, but for this function, we need
        # to get them from the object's last command.
        # Here we assume self.control_inputs is set by an external update.
        U1 = self.control_inputs[0]  # Total Thrust (in Body Z)
        torques = self.control_inputs[1:4] # Torques (in Body X, Y, Z)

        # 3. Translational Dynamics (in World Frame)
        # Acceleration = (1/m) * (R * Thrust_body) - Gravity_world
        thrust_world = R @ np.array([0, 0, U1])
        accel = (1.0 / self.m) * thrust_world - np.array([0, 0, self.g])
        
        # 4. Rotational Dynamics (in Body Frame)
        # Euler's Equation: I * w_dot = Tau - w x (I * w)
        ang_vel_body = np.array([p, q, r])
        ang_accel = self.I_inv @ (torques - np.cross(ang_vel_body, self.I @ ang_vel_body))

        # 5. Kinematic Equations (Angular velocity to Euler rates)
        # phi_dot = p + tan(theta) * (q * sin(phi) + r * cos(phi))
        # theta_dot = q * cos(phi) - r * sin(phi)
        # psi_dot = (q * sin(phi) + r * cos(phi)) / cos(theta)
        
        # Transformation matrix W
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        t_theta = np.tan(theta)

        # Avoid singularity at tan(pi/2)
        if abs(c_theta) < 1e-6:
            c_theta = 1e-6 * np.sign(c_theta)

        W = np.array([
            [1, s_phi * t_theta, c_phi * t_theta],
            [0, c_phi, -s_phi],
            [0, s_phi / c_theta, c_phi / c_theta]
        ])
        
        angle_rates = W @ ang_vel_body

        # 6. Assemble state derivative
        state_dot = np.zeros(12)
        state_dot[0:3] = vel
        state_dot[3:6] = accel
        state_dot[6:9] = angle_rates
        state_dot[9:12] = ang_accel

        return state_dot

    def update(self, dt, control_inputs):
        """
        Updates the quadcopter state over a time step dt.
        
        Args:
            dt (float): Time step (s)
            control_inputs (np.array[4]): [U1, U2, U3, U4]
                                          U1 = Total Thrust (N)
                                          U2 = Roll Torque (Nm)
                                          U3 = Pitch Torque (Nm)
                                          U4 = Yaw Torque (Nm)
        """
        self.control_inputs = control_inputs
        
        # *** NEW FIX: Use simple Euler integration ***
        # The adaptive solver solve_ivp was getting stuck on the
        # "stiff" equations from high initial PID gains.
        # A simple Euler step will not freeze and is fine for dt=0.01.
        
        state_dot = self._state_derivative(0, self.state)
        self.state = self.state + state_dot * dt

        # *** OLD FREEZING CODE ***
        # from scipy.integrate import solve_ivp
        # sol = solve_ivp(
        #     fun=self._state_derivative,
        #     t_span=[0, dt],
        #     y0=self.state,
        #     method='RK45'
        # )
        # self.state = sol.y[:, -1]
