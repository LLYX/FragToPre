import numpy as np

class MassSpecDataPoint:
    def __init__(self, mz, dt, rt, intensity):
        self.mz = mz
        self.dt = dt
        self.rt = rt
        self.intensity = intensity

    def __eq__(self, other):
        """Overrides the default implementation for equality."""
        if isinstance(other, MassSpecDataPoint):
            return (self.mz == other.mz and
                self.dt == other.dt and
                self.rt == other.rt and
                self.intensity == other.intensity)
        return False

    def get_array(self):
        return np.array([self.mz, self.dt, self.rt, self.intensity])

class MassSpecDataPointCloud:
    def __init__(self, data):
        self.data = data
        self.points_array = np.array([])

    def get_data(self):
        return np.array(self.data)

    def set_data(self, data):
        self.data = data
        
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

    def get_mz_dt_and_rt_bounded_points_array(self,
                                              lower_left,
                                              upper_right):
        pts = self.get_points_array()
        in_idx = np.all(
            (lower_left <= pts) & (pts <= upper_right), axis=1)
        in_box = pts[in_idx]

        return in_box
