import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import os


class IMUTracking:

    def __init__(self, data, freq):
        self.data = data
        self.freq = freq

    def data_add(self, data):
        self.data = self.data.append(data, ignore_index=True)
        # print(self.data)

    def visualize(self):
        print(self.data)

        earth_linear = self.earth_linear()

        ts = int(time.time())
        self.transform_plot(earth_linear, ts)

        dt = self.freq
        x, y, z = self.integrate(earth_linear[:, 0], earth_linear[:, 1], earth_linear[:, 2], dt)

        self.plot_3d(x, y, z, ts)

    def earth_linear(self):
        pitch = self.data.iloc[:, 3]
        roll = self.data.iloc[:, 4]
        yaw = self.data.iloc[:, 5]

        earth_linear = np.empty([self.data.shape[0], 3])
        for i in range(self.data.shape[0]):
            R_x, R_y, R_z = self.body_frame_rotation(pitch[i], roll[i], yaw[i])
            earth_linear[i, :] = R_z @ R_y @ R_x @ self.data.iloc[i, 0:3]

        return earth_linear

    @staticmethod
    def body_frame_rotation(x, y, z):
        R_x = np.array([[1, 0, 0],
                         [0, np.cos(-x), -np.sin(-x)],
                         [0, np.sin(-x), np.cos(-x)]])

        R_y = np.array([[np.cos(-y), 0, -np.sin(-y)],
                         [0, 1, 0],
                         [np.sin(-y), 0, np.cos(-y)]])

        R_z = np.array([[np.cos(-z), -np.sin(-z), 0],
                         [np.sin(-z), np.cos(-z), 0],
                         [0, 0, 1]])

        return R_x, R_y, R_z

    @staticmethod
    def integrate(x_ax, y_ax, z_ax, dt):
        x = cumtrapz(cumtrapz(x_ax, dx=dt), dx=dt)
        y = cumtrapz(cumtrapz(y_ax, dx=dt), dx=dt)
        z = cumtrapz(cumtrapz(z_ax, dx=dt), dx=dt)

        return x, y, z

    def transform_plot(self, earth_linear, ts):
        display = pd.DataFrame()
        display['LINEAR ACCELERATION X'] = self.data.iloc[:, 0]
        display['LINEAR ACCELERATION Y'] = self.data.iloc[:, 1]
        display['LINEAR ACCELERATION Z'] = self.data.iloc[:, 2]
        display['EARTH LINEAR ACCELERATION X'] = earth_linear[:, 0]
        display['EARTH LINEAR ACCELERATION Y'] = earth_linear[:, 1]
        display['EARTH LINEAR ACCELERATION Z'] = earth_linear[:, 2]

        cols_body = ['LINEAR ACCELERATION X', 'LINEAR ACCELERATION Y', 'LINEAR ACCELERATION Z', ]
        cols_earth = ['EARTH LINEAR ACCELERATION X', 'EARTH LINEAR ACCELERATION Y', 'EARTH LINEAR ACCELERATION Z']

        bodyplot = display.plot(y=cols_body, subplots=True, sharex=True, figsize=(20, 20), layout=(3, 1),
                                title=cols_body,
                                style='k', alpha=0.5)
        display.plot(y=cols_earth, subplots=True, layout=(3, 1), ax=bodyplot, sharex=True, style='g')

        plt.suptitle('Body Frame to Earth Frame Accelerations', fontsize=30)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        dir_name = './logs/result_plots/' + str(ts)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(dir_name + '/transform_plot.jpg', dpi=300)
        plt.show()

    @staticmethod
    def plot_3d(x, y, z, ts):
        fig,ax = plt.subplots(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, c='red', lw=5, label='phone trajectory')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')

        fig.suptitle('3D Trajectory of phone', fontsize=30)
        plt.tight_layout()
        plt.subplots_adjust(top=0.91)

        dir_name = './logs/result_plots/' + str(ts)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(dir_name + '/3d_plot.jpg', dpi=300)
        plt.show()
