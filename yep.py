'''implementing LQG controller on crazyflie drone'''
import logging
import time
import csv
from numpy import genfromtxt
import numpy as np
from scipy.linalg import solve_continuous_are

import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

logging.basicConfig(level=logging.ERROR)

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT = 0.5

class PathPlanner:
    '''decide which height to aim for next'''
    def __init__(self, z_pos):
        self.z_pos = z_pos
        self.look_ahead_radius = 0.05
        self.waypoints = np.zeros([100])
        self.goal = 0

        self.get_waypoints()

    def get_waypoints(self):
        '''get waypoints from csv'''
        self.waypoints = genfromtxt('ref-traj.csv', delimiter=',')
        self.find_closest_next_waypoint()

    def find_closest_next_waypoint(self):
        '''find closest next waypoint from imported csv'''
        dist = (self.waypoints - self.z_pos)**2
        i_closest_wp = np.argmin(dist)
        next_wp = self.waypoints[i_closest_wp + 1]
        self.goal = next_wp
        # dist_to_next = np.sqrt(next_wp - self.z_pos)**2
        # t_val = self.look_ahead_radius / dist_to_next
        # self.goal = (1 - t_val)*self.z_pos + t_val * next_wp

class LQRController:
    '''controller class'''
    def __init__(self, z_pos, z_goal):
        self.A = np.array([[0]])
        self.B = np.array([[1]])
        self.C = np.array([[1]])
        self.D = np.array([[0]])

        self.Q = np.diag([1])
        self.R = np.array([[0.1]])

        # Compute the LQR gain using the algebraic Riccati equation
        S = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ S

        self.z_pos = z_pos
        self.z_goal = z_goal

        self.z_vel = self.control()

    def control(self):
        # Compute the control input using the LQR gain and the current state
        print(self.z_goal - self.z_pos)
        return self.K * (self.z_goal - self.z_pos)


def log_pos_callback(_, data, __):
    '''fetch z pos'''

    global Z

    Z = data['range.zrange'] / 1000 #  z ranger is in mm for some reason

def write_to_csv(data):
    '''a function to log data to csv'''
    with open('file.csv', 'w', newline='', encoding='utf-8') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data)

def main(scf):
    '''main function'''

    with MotionCommander(scf, default_height=0.1) as mc:

        Z_list = []

        endtime = time.time() + 10

        while time.time() < endtime:

            z_pos = Z
            print(f'current alt is: {z_pos}')

            z_goal = PathPlanner(z_pos).goal
            print(f'goal alt is: {z_goal}')

            actuation = LQRController(z_pos, z_goal)
            print(f'velocity actuation is: {actuation.z_vel}')

            mc.start_linear_motion(0, 0, float(actuation.z_vel))

            print("-------------")

            Z_list.append(Z)

            time.sleep(0.1)

        mc.stop

    return Z_list

if __name__ == '__main__':

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        logconf = LogConfig(name='Range', period_in_ms=100)
        logconf.add_variable('range.zrange', 'float')

        cf = scf.cf
        cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(log_pos_callback)

        logconf.start()

        output = main(scf)

        logconf.stop()

        with open('file.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(output)
