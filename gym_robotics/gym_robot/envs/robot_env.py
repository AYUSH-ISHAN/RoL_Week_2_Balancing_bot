from math import *
import numpy as np
import pybullet as p
import pybullet_data
import gym
from gym import spaces
from gym.utils import seeding


class RoBots(gym.Env):

    def __init__(self):

        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.obv = []
        self.obvservation_space = spaces.Box(low=np.array([-pi, -pi, -5]),
                                             # also try to take -4pi/3 as lower in orientation part
                                             high=np.array([pi, pi, 5]))
        # dtype=np.float32)  # take a look at shape

        self.action_space = spaces.Discrete(9)  # keep odd values between 3 < # values < 15
        self.seeding_value()
        self.reset()

    def seeding_value(self, seed=None):
        # might not be useful as robot is to spawn at definite location
        # but may be useful in advanced cases to respawn at different location or orientation..
        # try such method after the Summer Camp.
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.Velocity_finder(action)
        p.stepSimulation()
        self.obv = self.prepare_observation()
        reward = self.Reward()  # also try a new way by making reward as a function of action.
        done = self.done_calculator()
        self.env_counter += 1

        return np.array(self.obv), reward, done, {}

    def reset(self):

        self.env_counter = 0
        self.velocity = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0, 0, 0.001]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        # path = os.path.abspath(os.path.dirname(__file__))
        self.botId = p.loadURDF("robot.urdf", cubeStartPos, cubeStartOrientation)
        self.obv = self.prepare_observation()

        return np.array(self.obv)

    def render(self, mode='human'):
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0, 0, 2]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        Turtle = p.loadURDF("robot.urdf", cubeStartPos, cubeStartOrientation)

    def done_calculator(self):
        pos, _ = p.getBasePositionAndOrientation(self.botId)
        # _, orien = p.getBase....(self.botId) then take out rotation along x and keep a parameter.
        return pos[2] < 0.2 or self.env_counter >= 1200  # they were 0.15 and 1500.

    def Velocity_finder(self, action):
        # two ways either make velocity a CONTINUOUS FUNCTION or keep a list of DISCRETE values.
        # For simpilcity, I took it as Discrete values.
        # velocity will be chosen from a list of velocities

        # DISCRETE METHOD of PID types:0

        para = 0.1  # adjustable parameter.
        vel = [-10.5 * para, -5.5 * para, -0.15 * para, 0., 0.15 * para, 5.5 * para, 10.5 * para][action]
        # to try - gap of 4 or 6 on basis of perfromance but there should be small
        # velocities to get balance in small deflection.
        velo = self.velocity + vel
        self.velocity = velo
        p.setJointMotorControl2(self.botId, jointIndex=0, controlMode=p.VELOCITY_CONTROL, targetVelocity=velo)
        p.setJointMotorControl2(self.botId, jointIndex=1, controlMode=p.VELOCITY_CONTROL, targetVelocity=velo)

    def Reward(self):

        # again, two ways make it discrete or continuous function of angle of rotation.

        # DISCRETE APPROACH
        _, orien = p.getBasePositionAndOrientation(self.botId)
        orien_euler = p.getEulerFromQuaternion(orien)
        # if orien_euler[0] == 0.0:
        #     return 500
        # elif abs(orien_euler[0]) < 0.1:
        #     return 100
        # elif 0.1 <= abs(orien_euler[0]) <= 0.2:
        #     return 0
        # else:
        #     return -100

        # CONTINUOUS APPROACH
        # it should be symmetric function i.e. same reward for positive theta
        # or velocity and negative theta or velocity - cosine or absolute function.
        # it should be inversely proportional to angle tilted.
        # it should be inversely proportional to velocity of turle's base.

        # Method 1:
        # return (10 / abs(orien_euler[0])) + (2 / abs(self.velocity))

        # Mehod 2:
        ## more priority is angle should not be tilted much
        param1 = 0.1
        param2 = 0.01  ### CLIPPING REWARDS: for faster learning   #####
        return (1 - abs(orien_euler[0])) * param1 - abs(self.velocity) * param2

    def prepare_observation(self):
        # obervation is (angle rotated, angular velocity of bot about axis through wheels, velocity of bot).
        Turtle_pos, Turtle_orn = p.getBasePositionAndOrientation(self.botId)
        Turtle_orn_euler = p.getEulerFromQuaternion(Turtle_orn)
        # getting the velocity and angular velocity part
        Base_vel, Base_ang_vel = p.getBaseVelocity(self.botId)
        obv_orn = Turtle_orn_euler[0]
        obv_ang_vel = Base_ang_vel[0]
        # obv_vel = Base_vel[0]
        return [obv_orn, obv_ang_vel, self.velocity]
