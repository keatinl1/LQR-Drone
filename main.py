'''implementing LQG controller on crazyflie drone'''
import logging
import time
import csv
import cflib.crtp

from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from controller import LQRController
from estimator import KalmanFilter
from planner import PathPlanner

logging.basicConfig(level=logging.ERROR)

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT = 0.5

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

        prev_vel = 0

        endtime = time.time() + 5

        while time.time() < endtime:

            # State Estimate
            raw_z = Z
            z_pos = KalmanFilter().update(raw_z, prev_vel) 
            print(f'current alt is: {z_pos}')

            # Path Plan (Tracking)
            z_goal = PathPlanner(z_pos).goal
            print(f'goal alt is: {z_goal}')

            # Control
            actuation = LQRController(z_pos, z_goal)
            print(f'velocity actuation is: {actuation.z_vel}')

            mc.start_linear_motion(0, 0, float(actuation.z_vel))

            print("-------------")

            Z_list.append(z_pos)

            prev_vel = float(actuation.z_vel)

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

        with open('file.csv', 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(output)
