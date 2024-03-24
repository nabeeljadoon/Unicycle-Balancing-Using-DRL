#!/usr/nabeel/Anaconda Env_Unicycle
# -*- coding: utf-8 -*-
## Unicycle Environment Development by Nabeel 

import pybullet as p
import pybullet_data
import numpy as np


class Unicycle:
    """
    unicycle class: The unicycle.py program is a Python script that contains a class definition for a unicycle model. The Unicycle class takes several parameters during initialization, such as whether to render the simulation, whether to use continuous or discrete actions, the time step of the simulation, the standard deviation of the initial position and velocity, the height of the ground, the radius of the ground, and whether to get an image of the environment.

The _build_model method is responsible for creating the unicycle model in the PyBullet physics engine. It creates the collision and visual shapes for each part of the unicycle, and sets up the joints between the different parts of the unicycle.

The img_from_state method is used to generate an image of the environment from a given state. It sets the camera position and orientation to look at the unicycle from a certain distance, and then computes the camera view and projection matrices. It then uses these matrices to capture an image of the environment.

The torque_from_action method is used to convert a given action into torque values for the two wheels. It takes a one-hot encoded action as input and returns the corresponding torque values.

The get_state method is used to get the current state of the unicycle. It returns the position, orientation, velocity, and angular velocity of the center column, as well as the angular velocity of the two wheels. It also checks if the unicycle has fallen down or moved out of bounds, and sets the done flag accordingly.

The reset method is used to reset the unicycle to its initial position and orientation. It sets the position and orientation of the center column and wheels to a random value, and resets the velocities and angular velocities to zero.

The continuous_torque method is used to convert a continuous action into torque values for the two wheels. It scales the input values to a suitable range and adds a constant offset to ensure that the wheels are always rotating in the positive direction.

The step method is used to simulate a single step of the unicycle. It takes an action as input and applies the corresponding torque to the wheels. It then simulates a single step of the physics engine and returns the new state of the unicycle, as well as the reward and done flag.

There are also several other methods and functions defined in the file, but they are not currently being used in the program.

Overall, the unicycle.py program provides a flexible and customizable interface for simulating a unicycle in the PyBullet physics engine. It can be used for a variety of tasks, such as reinforcement learning, control, and motion planning.

    """
    def __init__(self, render=False, continuous=False, time_step=1./30, sigma=0.2, down=1.0, floor_r=7, get_image=False):
        self.time_step = time_step
        self.sigma = sigma
        self.down = down
        self.render = render
        self.floor_r = floor_r
        self.get_image = get_image
        self.continuous = continuous
        if get_image:
            self.s_size = 120 * 160
        else:
            self.s_size = 19
        self._build_model()

    def _build_model(self):

        if self.render:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # disable rendering during creation
            cameraDistance = 6 # at 11m distance
            cameraYaw = 25
            cameraPitch = -45  # -30 deg (down angle)
            cameraTargetPosition = [0, 1, 1]  # focusing (0, 0, 1)
            p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

            # add camera follows the unicycle let an agent could get visual state

            # draw ring of radius = 9 m
            # n_polygon = 40
            # point_x = self.floor_r * np.cos(2 * np.pi * np.arange(n_polygon) / n_polygon)
            # point_y = self.floor_r * np.sin(2 * np.pi * np.arange(n_polygon) / n_polygon)
            # points = list(zip(point_x, point_y, np.ones(n_polygon)/20))
            # points_end = [points[(i + 1) % n_polygon] for i in range(n_polygon)]
            #
            # for i in range(n_polygon):
            #     p.addUserDebugLine(points[i], points_end[i], lineColorRGB=[0.5, 0.5, 0.05], lineWidth=3)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # Define ground area
        p.addUserDebugLine([-7, -7, 1], [-7, 7, 1], lineColorRGB=[0.7, 0.7, 0.07], lineWidth=50)
        p.addUserDebugLine([-7, 7, 1], [7, 7, 1], lineColorRGB=[0.7, 0.7, 0.07], lineWidth=50)
        p.addUserDebugLine([7, 7, 1], [7, -7, 1], lineColorRGB=[0.7, 0.7, 0.07], lineWidth=50)
        p.addUserDebugLine([7, -7, 1], [-7, -7, 1], lineColorRGB=[0.7, 0.7, 0.07], lineWidth=50)

        p.setRealTimeSimulation(enableRealTimeSimulation=0)  # p.TORQUE_CONTROL works only at non-real-time sim mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optional. to load preexisting data
        p.setGravity(0, 0, -9.8)  # g= 9.8 kg m / s^2

        p.setTimeStep(self.time_step)

        planeId = p.loadURDF("plane.urdf")
        #planeId = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

        C_Radius = 0.5  # Radius of wheels = 50 cm
        C_Height = 0.015   # width of wheels = 1 cm

        # Building blocks
        colCylinderId = p.createCollisionShape(p.GEOM_CYLINDER, radius=C_Radius, height=C_Height)  # Wheels
        colLodId = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=1.5)  # center column
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)  # indicator to check that wheels are rolling

        # link information [upper wheel, lower wheel, wheel indicator, wheel indicator]
        link_Masses = [3, 3, 0, 0]  # mass of the Wheels = 3 kg
        linkCollisionShapeIndices = [colCylinderId, colCylinderId, colSphereId, colSphereId]
        linkVisualShapeIndices = [1, -1, -1, -1]
        linkPositions = [[0, 0, 1.5 / 2], [0, 0, -1.5 / 2], [0, 0.3, 0], [0, 0.3, 0]]
        #linkOrientations = [[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
        linkOrientations = [[0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
        indices = [0, 0, 1, 2]  # linked to [ Lod, Lod, upper wheel, lower wheel ]
        jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_FIXED, p.JOINT_FIXED]  # rolling,rolling,fixed,fixed
        axis = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

        # position and orientation of the center of column
        basePosition = [0, 0, 0.5 + 1.5/2 + 1e-10]
        baseOrientation = [0, 0, 0, 1]

        visualShapeId = -1

        self.unicycleUid = \
            p.createMultiBody(6, colLodId,
                              visualShapeId,
                              basePosition,
                              baseOrientation,
                              linkMasses=link_Masses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=indices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis
                              )

        #p.changeDynamics(self.unicycleUid, -1, spinningFriction=0.0, rollingFriction=0.0, lateralFriction=0.02, linearDamping=0.001, angularDamping=0.001)

        # indices of "moving" joint
        self.jointinfo = []
        for i in range(p.getNumJoints(self.unicycleUid)):
            if p.getJointInfo(self.unicycleUid, i)[2] == 0:
                self.jointinfo.append(i)

        p.changeDynamics(self.unicycleUid, linkIndex=self.jointinfo[0], angularDamping=0.7)
    #
    # def set_init(self, position_variation=0, velocity_variation=0, orientation_variation=0):
    #     return np.random.normal(0,[position_variation, velocity_variation, orientation_variation])

    def img_from_state(self, state):
        x, y = state[:2]
        view = p.computeViewMatrix(cameraTargetPosition=[x, y, 1.],
                                   cameraEyePosition=[x + 3, y + 3., 3],
                                   cameraUpVector=[0, 0, 1])
        project = p.computeProjectionMatrixFOV(40, 4/3, 0.01, 30)
        img = p.getCameraImage(160, 120, viewMatrix=view, projectionMatrix=project)
        rgb = np.array(img[2])
        return np.ndarray.flatten(np.mean(rgb, -1))

    def torque_from_action(self, action):
        # one-hot
        action0 = [0, 0, 0, 0]
        action0[action] = 1.

        # torque = [torque on upper wheel, torque on lower wheel]
        # 32., 49. are experimentally chosen
        torque = [35. * (action0[0] - action0[1]),
                  50. * (action0[2] - action0[3])]
        return torque

    # return the present state
    def get_state(self):
        # state of the center of column
        center_position, center_angle = p.getBasePositionAndOrientation(self.unicycleUid)
        center_velocity, center_angular_velocity = p.getBaseVelocity(self.unicycleUid)

        # state of joint points (only get angular velocities of links)bn
        link_states = []
        for i in self.jointinfo:
            link_states.append(p.getLinkState(self.unicycleUid,
                                              linkIndex=i,
                                              computeLinkVelocity=1)[7])
        state = np.concatenate((center_position,
                                center_angle,
                                center_velocity,
                                center_angular_velocity,
                                link_states[0],
                                link_states[1]),
                               axis=None)

        # fall down or out of ground ends this episode
        out_of_ground = (np.abs(state[0]) > self.floor_r) or (np.abs(state[1]) > self.floor_r)
        fall_down = state[2] < self.down or state[2] > 1.5
        done = fall_down or out_of_ground

        reward = 1
        if done:  # Penalty
            reward = 0

        if self.get_image:
            return self.img_from_state(state.flatten()), done, reward
        else:
            return state.flatten(), done, reward

    # initialize the position with random normal distribution
    def reset(self):
        p.resetBasePositionAndOrientation(self.unicycleUid, [0, 0, 1.5 / 2. + 0.5 + 1e-10],
                                          [np.random.normal(0., self.sigma), np.random.normal(0., self.sigma), 0, 1])
        p.resetBaseVelocity(self.unicycleUid, [0, 0, 0], [0, 0, 0])
        for i in self.jointinfo:
            p.resetJointState(self.unicycleUid, i, 0)

        state, _, _ = self.get_state()
        return state

    def continuous_torque(self, action):
        t1, t2 = action[0]
        t1 = 4. * t1
        t2 = 20. * t2
        t1 = t1 + np.sign(t1) * 30.
        t2 = t2 + np.sign(t2) * 30.
        return [t1, t2]

    # play one step
    def step(self, action):
        if self.continuous:
            torque = self.continuous_torque(action)
        else:
            torque = self.torque_from_action(action)

        p.setJointMotorControlArray(self.unicycleUid,
                                    self.jointinfo,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=torque)

        # should add print angular velocity(acceleration) for given torque depends on time_step
        p.stepSimulation()
        state, done, reward = self.get_state()

        return state, reward, done, []  # last return slot for debug


if __name__ == '__main__':
    print(p.isNumpyEnabled())

