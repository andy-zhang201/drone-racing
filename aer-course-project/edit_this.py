"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates 
while considering the uncertainty of the obstacles. The students should show that the proposed 
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
import numpy as np

from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        self.iteration = 0

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles. 
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        ## visualization
        # Plot trajectory in each dimension and 3D.
        # plot_trajectory(t_scaled, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


    def planning(self, use_firmware, initial_info):
        """Trajectory planning algorithm"""

        # Gate Coords Order
        gate1 = self.NOMINAL_GATES[0]
        gate2 = self.NOMINAL_GATES[1]
        gate3 = self.NOMINAL_GATES[2]
        gate4 = self.NOMINAL_GATES[3]

        ### INPUT ORDER OF GATES HERE ###
        # gates_ordered = [gate3,gate1,gate2,gate4]

        gates_ordered = [gate1,gate3,gate2,gate4]

        waypoints_x, waypoints_y, splits = ecu.make_plan(self.initial_obs[0],self.initial_obs[2], gates_ordered)
        splits.insert(0,0)

        waypoints = []
        height = initial_info["gate_dimensions"]["tall"]["height"]
        for i in range(len(waypoints_x)):
            waypoints.append([waypoints_x[i], waypoints_y[i], height])

        print(waypoints)   

        # How to trajectory plan better:
        """
        1. split waypoints list into 4 subproblems
        2. for each sublist, create variables t, fx, fy, fz using np.arange and poly1d.
        3. Set a duration of flight for each sublist.
        4. Concatenate all t sublists into a signle one for t_scaled
        5. Concatenate all sub self.ref_x = fx, self.ref_y = fy, self.ref_z = fz
        """
        print(f"Waypoints: {waypoints}")
        print(f"Waypoints length: {len(waypoints)}")

        self.waypoints = np.array(waypoints)
        self.ref_x = np.array([])
        self.ref_y = np.array([])
        self.ref_z = np.array([])
        t = np.arange(self.waypoints.shape[0])
        duration = 20
        t_scaled=[]

        # new_splits = [(0,splits[0])]
        # for idx in splits[1:]:
        #     new_splits.append((idx))
        

        for idx in splits[:-1]:
            print(f"Split at {idx}")
            temp_waypoints = np.array(waypoints[idx:splits[splits.index(idx)+1]-1])

            if idx == splits[-2]:
                print('Last split')
                temp_waypoints = np.array(waypoints[idx:])

            deg = int(len(temp_waypoints)/2)
            t_temp = np.arange(temp_waypoints.shape[0])
            temp_fx = np.poly1d(np.polyfit(t_temp, temp_waypoints[:,0], deg))
            temp_fy = np.poly1d(np.polyfit(t_temp, temp_waypoints[:,1], deg))
            temp_fz = np.poly1d(np.polyfit(t_temp, temp_waypoints[:,2], deg))

            temp_duration = duration*len(temp_waypoints)/len(waypoints)

            t_scaled_temp = np.linspace(t_temp[0], t_temp[-1], int(temp_duration*self.CTRL_FREQ))

            self.ref_x = np.concatenate((self.ref_x, temp_fx(t_scaled_temp)))
            self.ref_y = np.concatenate((self.ref_y, temp_fy(t_scaled_temp)))
            self.ref_z = np.concatenate((self.ref_z, temp_fz(t_scaled_temp)))
            t_scaled.append(t_scaled_temp)

            # # Append gate waypoints
            # self.ref_x = np.concatenate((self.ref_x, [waypoints[-1][0]]))
            # self.ref_y = np.concatenate((self.ref_y, [waypoints[-1][1]]))
            # self.ref_z = np.concatenate((self.ref_z, [waypoints[-1][2]]))
            # t_scaled_temp = np.linspace(t_scaled[-1], t_scaled[-1] + 2, int(2*self.CTRL_FREQ))
            # t_scaled.append(t_scaled_temp)


            # self.ref_x
        # Append last waypoint
        self.ref_x = np.concatenate((self.ref_x, [waypoints[-1][0]]))
        self.ref_y = np.concatenate((self.ref_y, [waypoints[-1][1]]))
        self.ref_z = np.concatenate((self.ref_z, [waypoints[-1][2]]))
        t_scaled_temp = np.linspace(t_scaled[-1], t_scaled[-1] + 2, int(2*self.CTRL_FREQ))
        t_scaled.append(t_scaled_temp)

        # deg = len(self.waypoints) - 18
        # t = np.arange(self.waypoints.shape[0])
        # fx = np.poly1d(np.polyfit(t, self.waypoints[:,0], deg))
        # fy = np.poly1d(np.polyfit(t, self.waypoints[:,1], deg))
        # fz = np.poly1d(np.polyfit(t, self.waypoints[:,2], deg))
        # duration = 13
        # print(f'T: {len(t)}')
        # t_s1 = np.linspace(t[0], t[10], int(duration*self.CTRL_FREQ))
        # t_s2 = np.linspace(t[10], t[-1], int(duration*self.CTRL_FREQ*2))
        # # t_scaled = np.linspace(t[0], t[-1], int(duration*self.CTRL_FREQ))
        # t_scaled = np.concatenate((t_s1, t_s2))
        # #print(t_scaled)
        # print("This is fx:")
        # print(fx(t_scaled))

        # self.ref_x = fx(t_scaled)
        # self.ref_y = fy(t_scaled)
        # self.ref_z = fz(t_scaled)

        # self.ref_x = self.waypoints[:,0]
        # self.ref_y = self.waypoints[:,1]
        # self.ref_z = self.waypoints[:,2]



        #########################
        # REPLACE THIS (END) ####
        #########################

        return t_scaled

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        # [INSTRUCTIONS] 
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file) 
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration= int(time*self.CTRL_FREQ)

        if iteration > self.iteration+1:
            print(f"Missed iteration: {iteration-1}")
            iteration = self.iteration+1
        
        self.iteration = iteration
        print(f"Iteration: {iteration}")
        #########################
        # REPLACE THIS (START) ##
        #########################
        # Flight splits:
        cmdFullStateStart = 3
        duration = 20

        cmdFullStateEnd = cmdFullStateStart + duration

        landStart = cmdFullStateEnd + 3

        turnOffTime = landStart + 3

        # print("The info. of the gates ")
        # print(self.NOMINAL_GATES)

        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        # [INSTRUCTIONS] Example code for using cmdFullState interface   
        elif iteration >= cmdFullStateStart*self.CTRL_FREQ and iteration < cmdFullStateEnd*self.CTRL_FREQ:
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == cmdFullStateEnd*self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

        elif iteration == landStart*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == turnOffTime*self.CTRL_FREQ-1:
            command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)
            args =[]


        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project. 
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
