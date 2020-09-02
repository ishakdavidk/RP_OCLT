import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib; matplotlib.use("TkAgg")
from mpl_toolkits import mplot3d
import time
from datetime import datetime
import threading
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import proj3d
import sys

x_vals, y_vals, z_vals = [], [], []
fig = None
ax = None
acc_x_in, acc_y_in, acc_z_in = 0, 0, 0

data_gbl = pd.DataFrame()
filtered_data = pd.DataFrame()
earth_linear_gbl = np.empty([1, 3])

star = mpath.Path.unit_regular_star(3)
unit = mpath.Path.unit_regular_polygon(9)
verts = np.concatenate([unit.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([unit.codes, star.codes])
marker_style = mpath.Path(verts, codes)


class IMUTrackingLive:

    def __init__(self):
        self.pos_init = [0, 0, 0]
        self.vel_init = [0, 0, 0]

        self.calibrate_iter = 0
        self.acc_lower_bound = [0, 0, 0]
        self.vel_lower_bound = [0, 0, 0]

        self.time_init = int(round(time.time() * 1000))

        self.anim = None
        self.trans_anim = None

        self.raw_data_bhpf = pd.DataFrame()

        anim_thread = threading.Thread(target=self.run_anim)
        anim_thread.start()

    def visualize(self, data, data_time):
        global acc_x_in, acc_y_in, acc_z_in, raw_data_bhpf, data_gbl
        acc_x_in, acc_y_in, acc_z_in = data.iloc[0, 0], data.iloc[0, 1], data.iloc[0, 2]

        calibrate_n = 10
        self.calibrate_iter += 1

        data_cp = data.copy()

        self.raw_data_bhpf = self.raw_data_bhpf.append(data_cp, ignore_index=True)
        data_gbl = data_gbl.append(data_cp, ignore_index=True)

        if self.calibrate_iter > calibrate_n:
            self.compute(data, data_time)
            # compute_thread = threading.Thread(target=self.compute())
            # compute_thread.start()

    def compute(self, data, data_time):
        global earth_linear_gbl, data_gbl, filtered_data
        # print(data)

        data_ahpf = self.raw_data_bhpf.copy()
        # b, a = self.butter_highpass(0.5, 0.2, 5)
        b, a = self.butter_bandpass(0.02, 0.08, 0.2, 5)
        filtered_data = signal.filtfilt(b, a, data_ahpf.iloc[:, 0:3], axis=1, method="gust")

        earth_linear_acc = self.earth_linear(data)
        earth_linear_gbl = np.concatenate((earth_linear_gbl, earth_linear_acc))

        dt = self.time_subs(self.time_init, data_time)
        # print("dt : ", dt)
        self.time_init = data_time

        # integrate acceleration
        vel_current = self.integrate(earth_linear_acc, self.vel_init, dt)
        for i in range (len(self.vel_init)):
            self.vel_init[i] = vel_current[0, i]

        # integrate velocity
        pos_current = self.integrate(vel_current, self.pos_init, dt)
        for i in range(len(self.pos_init)):
            self.pos_init[i] = pos_current[0, i]

        global x_vals, y_vals, z_vals
        x_vals.append(pos_current[0, 0])
        y_vals.append(pos_current[0, 1])
        z_vals.append(pos_current[0, 2])

    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        high = (2*cutoff)/(1/fs)
        b, a = signal.butter(order, high, btype='high', analog=False)

        return a,b

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        return b, a

    def earth_linear(self, data):
        pitch = data.iloc[0, 3]
        roll = data.iloc[0, 4]
        yaw = data.iloc[0, 5]

        earth_linear = np.empty([1, 3])
        R_x, R_y, R_z = self.body_frame_rotation(pitch, roll, yaw)
        earth_linear[0, :] = R_z @ R_y @ R_x @ data.iloc[0, 0:3]

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
    def time_subs(timestamp1, timestamp2):
        timestamp1 = datetime.fromtimestamp(timestamp1 / 1000.0)
        timestamp2 = datetime.fromtimestamp(timestamp2 / 1000.0)

        subs = timestamp2 - timestamp1
        seconds = subs.total_seconds()

        return seconds

    @staticmethod
    def integrate(vals_in, vals_init, dt):
        vals_out = np.empty([1, 3])

        for i in range(3):
            vals_out[0, i] = vals_in[0, i] * dt + vals_init[i]

        return vals_out

    def animate(self):
        global x_vals, y_vals, z_vals, ax, fig, marker_style, acc_x_in, acc_y_in, acc_z_in

        plt.cla()
        ax = plt.axes(projection='3d')
        if x_vals:
            n = len(x_vals) - 1
            ax.plot3D([x_vals[n]], [y_vals[n]], [z_vals[n]], marker=marker_style, markersize=15,
                      c='blue', lw=5, label='phone trajectory')
            ax.plot3D(x_vals, y_vals, z_vals, linestyle=":",
                      c='red', lw=2)

            # label
            x_proj, y_proj, _ = proj3d.proj_transform(x_vals[n], y_vals[n], z_vals[n], ax.get_proj())
            label = "Acc ({:.4f},{:.4f},{:.4f})\nPos ({:.2f},{:.2f},{:.2f})"\
                .format(acc_x_in, acc_y_in, acc_z_in, x_vals[n], y_vals[n], z_vals[n])
            ax.annotate(label, (x_proj, y_proj), textcoords="offset points",
                         xytext=(0, 20), ha='center', size=14)

        ax.set_xlabel('x (m)', fontsize=15, labelpad=15)
        ax.set_ylabel('y (m)', fontsize=15, labelpad=15)
        ax.set_zlabel('z (m)', fontsize=15, labelpad=15)

        fig.suptitle('3D Trajectory of phone', fontsize=30)
        fig.tight_layout()
        plt.subplots_adjust(top=0.8)

    def animate_2(self):
        global bodyplot, data_gbl, earth_linear_gbl, fig_2, ax_2

        len_filtered_data = len(filtered_data)
        len_data_gbl = len(data_gbl)

        if len_filtered_data > 0:
            # print("len_filtered_data : ", len_filtered_data)
            # print("len_data_gbl ", len_data_gbl)

            display = pd.DataFrame()
            display['LINEAR ACCELERATION X'] = data_gbl.iloc[:len_filtered_data, 0]
            display['LINEAR ACCELERATION Y'] = data_gbl.iloc[:len_filtered_data, 1]
            display['LINEAR ACCELERATION Z'] = data_gbl.iloc[:len_filtered_data, 2]
            display['FILTERED LINEAR ACCELERATION X'] = filtered_data[:, 0]
            display['FILTERED LINEAR ACCELERATION Y'] = filtered_data[:, 1]
            display['FILTERED LINEAR ACCELERATION Z'] = filtered_data[:, 2]

            print(display.iloc[len(display)-1, :])

            cols_body = ['LINEAR ACCELERATION X', 'LINEAR ACCELERATION Y', 'LINEAR ACCELERATION Z', ]
            cols_earth = ['FILTERED LINEAR ACCELERATION X', 'FILTERED LINEAR ACCELERATION Y', 'FILTERED LINEAR ACCELERATION Z']

            plt.cla()
            bodyplot = display.plot(y=cols_body, subplots=True, sharex=True, figsize=(20, 20), layout=(3, 1),
                                    title=cols_body,
                                    style='k', alpha=0.5, ax=ax)
            display.plot(y=cols_earth, subplots=True, ax=bodyplot, style='g')

            fig.suptitle('Body Frame to Earth Frame Accelerations', fontsize=25)
            fig.tight_layout()
            plt.subplots_adjust(top=0.89)

    def run_anim(self):
        global fig, ax

        fig, ax = plt.subplots(figsize=(10, 10))
        # self.anim = FuncAnimation(fig, IMUTrackingLive.animate, interval=200)
        self.anim = FuncAnimation(fig, IMUTrackingLive.animate_2, interval=200)
        plt.show()


    def stop_anim(self):
        global x_vals, y_vals, z_vals
        x_vals.clear()
        y_vals.clear()
        z_vals.clear()

        self.anim.event_source.stop()
        # plt.close()



