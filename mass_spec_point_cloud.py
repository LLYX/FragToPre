import numpy as np
import pyopenms as ms

from scipy.spatial import ckdtree

class MassSpecDataPoint:
    def __init__(self, mz, rt, dt, intensity):
        self.mz = mz
        self.dt = dt
        self.rt = rt
        self.intensity = intensity

    def get_array(self):
        return np.array([self.mz, self.dt, self.rt, self.intensity])

class MassSpecDataPointCloud:
    def __init__(self, data):
        self.data = data
        self.points_array = np.array([])
        
    def get_points_array(self):
        if not self.points_array:
            self.points_array = np.array(
                [point.get_array() for point in self.data])

        return np.array(self.points_array)

    def get_mz_and_dt_bounded_points_array(self, lower_left, upper_right):
        xy_pts = self.get_points_array()[:, [0, 1]]
        in_idx = np.all(
            (lower_left <= xy_pts) & (xy_pts <= upper_right), axis=1)
        in_box = self.get_points_array()[in_idx]

        return in_box
