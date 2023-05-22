import numpy as np

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
        self.waypoints = np.genfromtxt('ref-traj.csv', delimiter=',')
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