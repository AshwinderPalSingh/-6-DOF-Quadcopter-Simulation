import numpy as np

class PID:
    """A simple PID controller class."""
    def __init__(self, Kp, Ki, Kd, setpoint=0, integral_limit=1.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self._integral = 0
        self._last_error = 0
        self.integral_limit = integral_limit
        self._first_update = True  # Prevents derivative kick

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        
        # Proportional
        P = self.Kp * error
        
        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        I = self.Ki * self._integral
        
        # Derivative
        if self._first_update or dt <= 0:
            error_dot = 0.0
            self._first_update = False
        else:
            error_dot = (error - self._last_error) / dt
        
        D = self.Kd * error_dot
        self._last_error = error
        
        return P + I + D

    def set_target(self, target):
        self.setpoint = target
        
    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._first_update = True

class Controller:
    """
    A cascaded PID Controller for 6-DOF Quadcopter.
    
    - Outer loop (Position): PID_x, PID_y, PID_z
    - Inner loop (Attitude): PID_roll, PID_pitch, PID_yaw
    """
    def __init__(self, quad_params):
        self.m = quad_params['m']
        self.g = quad_params['g']
        
        # --- GAINS --- (These are now stable)
        self.pid_gains = {
            # Position (Outer Loop)
            'x':   {'Kp': 1.0, 'Ki': 0.05, 'Kd': 1.0},
            'y':   {'Kp': 1.0, 'Ki': 0.05, 'Kd': 1.0},
            'z':   {'Kp': 2.0, 'Ki': 0.5,  'Kd': 2.0},
            
            # Attitude (Inner Loop)
            'roll':  {'Kp': 5.0, 'Ki': 0.1, 'Kd': 1.0},
            'pitch': {'Kp': 5.0, 'Ki': 0.1, 'Kd': 1.0},
            'yaw':   {'Kp': 2.0, 'Ki': 0.1, 'Kd': 1.0}
        }
        
        # --- Instantiate Controllers ---
        self.pid_x = PID(**self.pid_gains['x'])
        self.pid_y = PID(**self.pid_gains['y'])
        self.pid_z = PID(**self.pid_gains['z'])
        
        self.pid_roll = PID(**self.pid_gains['roll'])
        self.pid_pitch = PID(**self.pid_gains['pitch'])
        self.pid_yaw = PID(**self.pid_gains['yaw'])
        
        # Set integral limits (anti-windup)
        self.pid_x.integral_limit = np.deg2rad(15) 
        self.pid_y.integral_limit = np.deg2rad(15)
        
        # Set maximum torque limits
        self.max_torque_roll_pitch = 1.0  # (Nm)
        self.max_torque_yaw = 0.5         # (Nm)
        
    def set_target(self, pos_target, yaw_target):
        self.pid_x.set_target(pos_target[0])
        self.pid_y.set_target(pos_target[1])
        self.pid_z.set_target(pos_target[2])
        self.pid_yaw.set_target(yaw_target)

    def update(self, state, dt):
        # Unpack state
        pos = state[0:3]
        vel = state[3:6]
        angles = state[6:9]
        ang_vel = state[9:12]
        
        phi, theta, psi = angles
        p, q, r = ang_vel

        # === OUTER LOOP (Position Control) ===
        # Calculate desired thrust, roll, and pitch

        # 1. Z-Controller (Altitude)
        thrust_cmd = self.pid_z.update(pos[2], dt)
        
        # *** THIS IS THE CRITICAL FIX ***
        # 2. X-Controller (Position)
        # Positive X error requires a POSITIVE pitch (theta)
        theta_des = self.pid_x.update(pos[0], dt)
        
        # 3. Y-Controller (Position)
        # Positive Y error requires a NEGATIVE roll (phi)
        phi_des = -self.pid_y.update(pos[1], dt)
        # *** END OF FIX ***

        # Limit desired tilt
        max_tilt = np.deg2rad(20)
        theta_des = np.clip(theta_des, -max_tilt, max_tilt)
        phi_des = np.clip(phi_des, -max_tilt, max_tilt)

        # === INNER LOOP (Attitude Control) ===
        # Calculate desired torques based on attitude errors

        # 1. Roll-Controller
        self.pid_roll.set_target(phi_des)
        U2 = self.pid_roll.update(phi, dt)
        
        # 2. Pitch-Controller
        self.pid_pitch.set_target(theta_des)
        U3 = self.pid_pitch.update(theta, dt)
        
        # 3. Yaw-Controller (Target is set externally)
        U4 = self.pid_yaw.update(psi, dt)
        
        # Clamp torque outputs
        U2 = np.clip(U2, -self.max_torque_roll_pitch, self.max_torque_roll_pitch)
        U3 = np.clip(U3, -self.max_torque_roll_pitch, self.max_torque_roll_pitch)
        U4 = np.clip(U4, -self.max_torque_yaw, self.max_torque_yaw)
        
        # === MIXING AND COMPENSATION ===
        
        # Compensate thrust for gravity and attitude
        c_phi = np.cos(phi)
        c_theta = np.cos(theta)
        
        if abs(c_phi) < 0.1 or abs(c_theta) < 0.1:
            U1 = self.m * self.g
        else:
            U1 = (self.m * self.g + thrust_cmd) / (c_phi * c_theta)
            
        U1 = max(0.0, U1)

        # Return control inputs [U1, U2, U3, U4]
        return np.array([U1, U2, U3, U4])
