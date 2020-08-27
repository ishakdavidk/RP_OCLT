import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib; matplotlib.use("TkAgg")
from mpl_toolkits import mplot3d
import time
from datetime import datetime
import os
import threading

x_vals, y_vals, z_vals = [], [], []
fig = None
ax = None

class IMUTrackingLive:

    def __init__(self):
        self.pos_init_x, self.pos_init_y, self.pos_init_z = 0, 0, 0
        self.vel_init_x, self.vel_init_y, self.vel_init_z = 0, 0, 0

        self.time_init = int(round(time.time() * 1000))

        self.anim = None
        thread = threading.Thread(target=self.run_anim)
        thread.start()

    def visualize(self, data, data_time):
        # print(data)

        earth_linear = self.earth_linear(data)

        dt = self.time_subs(self.time_init, data_time)
        # print("dt : ", dt)
        self.time_init = data_time

        x, y, z = self.integrate(earth_linear[0, 0], earth_linear[0, 1], earth_linear[0, 2], dt)
        print(x, ", ", y, ", ", z)

        global x_vals, y_vals, z_vals
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

    def earth_linear(self, data):
        pitch = data.iloc[0, 3]
        roll = data.iloc[0, 4]
        yaw = data.iloc[0, 5]

        earth_linear = np.empty([1, 3])
        R_x, R_y, R_z = self.body_frame_rotation(pitch, roll, yaw)
        earth_linear[0, :] = R_z @ R_y @ R_x @ data.iloc[0, 0:3]

        return earth_linear

    @staticmethod
    def time_subs(timestamp1, timestamp2):
        timestamp1 = datetime.fromtimestamp(timestamp1 / 1000.0)
        timestamp2 = datetime.fromtimestamp(timestamp2 / 1000.0)

        subs = timestamp2 - timestamp1
        seconds = subs.total_seconds()
        return seconds

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

    def integrate(self, A_x, A_y, A_z, dt):
        V_x = A_x * dt + self.vel_init_x
        x = V_x * dt + self.pos_init_x
        self.vel_init_x = V_x
        self.pos_init_x = x

        V_y = A_y * dt + self.vel_init_y
        y = V_y * dt + self.pos_init_y
        self.vel_init_y = V_y
        self.pos_init_y = y

        V_z = A_z * dt + self.vel_init_z
        z = V_z * dt + self.pos_init_z
        self.vel_init_z = V_z
        self.pos_init_z = z

        return x, y, z

    def animate(self):
        global x_vals, y_vals, z_vals, ax, fig

        plt.cla()
        ax = plt.axes(projection='3d')
        ax.plot3D(x_vals, y_vals, z_vals, c='red', lw=5, label='phone trajectory')
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_zlabel('Z position (m)')

        fig.suptitle('3D Trajectory of phone', fontsize=30)
        plt.tight_layout()
        plt.subplots_adjust(top=0.91)

    def run_anim(self):
        global fig, ax

        fig, ax = plt.subplots(figsize=(10, 10))
        self.anim = FuncAnimation(fig, IMUTrackingLive.animate, interval=1000)
        plt.show()

    def stop_anim(self):
        self.anim.event_source.stop()
