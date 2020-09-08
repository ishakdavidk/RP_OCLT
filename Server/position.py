import pandas as pd
import numpy as np
from scipy import signal
import matplotlib; matplotlib.use("TkAgg")
import time
from datetime import datetime
import threading
import sys
import math


class Position:
    def __init__(self):
        self.earth_acc_iter = 0
        self.vel_iter = 0
        self.pos_iter = 0

        self.acc_lower_bound = [0, 0, 0]
        self.vel_lower_bound = [0, 0, 0]

        self.raw_earth_acc_vals = np.empty([1, 3])

        self.time_init = int(round(time.time() * 1000))

        self.vel_init = [0, 0, 0]
        self.raw_vel_vals = np.empty([1, 3])

        self.pos_init = [0, 0, 0]
        self.raw_pos_vals = np.empty([1, 3])

        # b, a = self.butter_highpass(0.03, 0.2, 5)

        # Earth linear acceleration filter coefficient vector
        self.acc_b_x, self.acc_a_x = self.butter_bandpass(1, 2, 5, 5)
        self.acc_b_y, self.acc_a_y = self.butter_bandpass(0.6, 2.4, 5, 5)
        self.acc_b_z, self.acc_a_z = self.butter_bandpass(1, 2.1, 5, 5)

        # Velocity filter coefficient vector
        self.vel_b_x, self.vel_a_x = self.butter_bandpass(1.5, 2.4, 5, 5)
        self.vel_b_y, self.vel_a_y = self.butter_bandpass(1.5, 2.4, 5, 5)
        self.vel_b_z, self.vel_a_z = self.butter_bandpass(1.5, 2.4, 5, 5)

        # Position filter coefficient vector
        self.pos_b_x, self.pos_a_x = self.butter_bandpass(1.5, 2, 5, 5)
        self.pos_b_y, self.pos_a_y = self.butter_bandpass(1.5, 2, 5, 5)
        self.pos_b_z, self.pos_a_z = self.butter_bandpass(1.5, 2, 5, 5)

        df_cols = ['LINEAR ACC X', 'LINEAR ACC Y', 'LINEAR ACC Z',
                   'EARTH LINEAR ACC X', 'EARTH LINEAR ACC Y', 'EARTH LINEAR ACC Z',
                   'FILTERED EARTH LINEAR ACC X', 'FILTERED EARTH LINEAR ACC Y', 'FILTERED EARTH LINEAR ACC Z',
                   'VEL X', 'VEL Y', 'VEL Z',
                   'FILTERED VEL X', 'FILTERED VEL Y', 'FILTERED VEL Z',
                   'POS X', 'POS Y', 'POS Z',
                   'FILTERED POS X', 'FILTERED POS Y', 'FILTERED POS Z']

        self.display = pd.DataFrame(columns=df_cols)

    def update(self, data, data_time):
        # Collect first 10 linear acceleration data
        calibrate_n = 10

        if self.earth_acc_iter > calibrate_n:
            if self.vel_iter > calibrate_n:
                if self.pos_iter > calibrate_n:
                    # Run main code
                    pos_current, dt = self.compute(data, data_time)
                    return pos_current, self.display, dt
                else:
                    # Collect first 10 position raw data
                    self.pos_iter += 1

                    # calculate filtered earth frame linear acc
                    earth_acc_raw, earth_acc = self.compute_filtered_earth_acc(data)

                    # calculate dt
                    dt = self.time_subs(self.time_init, data_time)
                    self.time_init = data_time

                    # calculate filtered velocity
                    vel_raw, vel = self.compute_filtered_vel(earth_acc, dt)

                    # integrate velocity
                    pos_raw = self.integrate(vel.copy(), self.pos_init, dt)
                    self.raw_pos_vals = np.concatenate((self.raw_pos_vals, pos_raw))
                    for i in range(len(self.pos_init)):
                        self.pos_init[i] = pos_raw[0, i]

                    empty_vals = np.zeros((1, 3))
                    self.display_update(data, earth_acc_raw, earth_acc, vel_raw, vel, pos_raw, empty_vals)

                    return empty_vals, self.display, dt
            else:
                # Collect first 10 velocity raw data
                self.vel_iter += 1

                # calculate filtered earth frame linear acc
                earth_acc_raw, earth_acc = self.compute_filtered_earth_acc(data)

                # calculate dt
                dt = self.time_subs(self.time_init, data_time)
                self.time_init = data_time

                # integrate acceleration
                vel_raw = self.integrate(earth_acc.copy(), self.vel_init, dt)
                self.raw_vel_vals = np.concatenate((self.raw_vel_vals, vel_raw))
                for i in range(len(self.vel_init)):
                    self.vel_init[i] = vel_raw[0, i]

                empty_vals = np.zeros((1, 3))
                self.display_update(data, earth_acc_raw, earth_acc, vel_raw, empty_vals, empty_vals, empty_vals)

                return empty_vals, self.display, dt
        else:
            # Collect first 10 earth linear acceleration raw data
            self.earth_acc_iter += 1

            # calculate earth frame linear acc
            earth_acc_raw = self.earth_linear(data.copy())
            self.raw_earth_acc_vals = np.concatenate((self.raw_earth_acc_vals, earth_acc_raw))

            # calculate dt
            dt = self.time_subs(self.time_init, data_time)
            self.time_init = data_time

            # update display data
            empty_vals = np.zeros((1, 3))
            self.display_update(data, earth_acc_raw, empty_vals, empty_vals, empty_vals, empty_vals, empty_vals)

            return empty_vals, self.display, dt

    def compute(self, data, data_time):
        # calculate filtered earth frame linear acc
        earth_acc_raw, earth_acc = self.compute_filtered_earth_acc(data)

        # calculate dt
        dt = self.time_subs(self.time_init, data_time)
        self.time_init = data_time
        # print("dt : ", dt)

        # calculate filtered velocity
        vel_raw, vel = self.compute_filtered_vel(earth_acc, dt)

        # integrate velocity
        pos_raw, pos = self.compute_filtered_pos(vel, dt)

        self.display_update(data, earth_acc_raw, earth_acc, vel_raw, vel, pos_raw, pos)

        return pos, dt

    def compute_filtered_earth_acc(self, data):
        earth_acc = self.earth_linear(data.copy())
        earth_acc_raw = earth_acc.copy()
        self.raw_earth_acc_vals = np.concatenate((self.raw_earth_acc_vals, earth_acc_raw))

        # band pass filter for earth linear acceleration
        earth_acc_bbpf = self.raw_earth_acc_vals.copy()
        filtered_earth_acc_x = signal.filtfilt(self.acc_b_x, self.acc_a_x, earth_acc_bbpf[:, 0], axis=0, method="gust")
        filtered_earth_acc_y = signal.filtfilt(self.acc_b_y, self.acc_a_y, earth_acc_bbpf[:, 1], axis=0, method="gust")
        filtered_earth_acc_z = signal.filtfilt(self.acc_b_z, self.acc_a_z, earth_acc_bbpf[:, 2], axis=0, method="gust")

        # print("Unfiltered Acc : ", data)
        len_filtered_earth_acc = len(filtered_earth_acc_x) - 1
        # print("Filter Result x : ", filtered_acc_x[len_filtered_acc_data], ", y : ", filtered_acc_y[len_filtered_acc_data], ", z : ", filtered_acc_z[len_filtered_acc_data])
        earth_acc[0, 0] = filtered_earth_acc_x[len_filtered_earth_acc]
        earth_acc[0, 1] = filtered_earth_acc_y[len_filtered_earth_acc]
        earth_acc[0, 2] = filtered_earth_acc_z[len_filtered_earth_acc]

        # for i in range(3):
        #     if -0.3 < earth_acc[0, i] < 0.3:
        #         earth_acc[0, i] = 0

        return earth_acc_raw, earth_acc

    def compute_filtered_vel(self, earth_acc, dt):
        # integrate acceleration
        vel = self.integrate(earth_acc.copy(), self.vel_init, dt)
        vel_raw = vel.copy()
        self.raw_vel_vals = np.concatenate((self.raw_vel_vals, vel_raw))

        # band pass filter for velocity
        vel_bbpf = self.raw_vel_vals.copy()
        filtered_vel_x = signal.filtfilt(self.vel_b_x, self.vel_a_x, vel_bbpf[:, 0], axis=0, method="gust")
        filtered_vel_y = signal.filtfilt(self.vel_b_y, self.vel_a_y, vel_bbpf[:, 1], axis=0, method="gust")
        filtered_vel_z = signal.filtfilt(self.vel_b_z, self.vel_a_z, vel_bbpf[:, 2], axis=0, method="gust")

        len_filtered_vel = len(filtered_vel_x) - 1
        vel[0, 0] = filtered_vel_x[len_filtered_vel]
        vel[0, 1] = filtered_vel_y[len_filtered_vel]
        vel[0, 2] = filtered_vel_z[len_filtered_vel]

        # for i in range(3):
        #     if -0.3 < vel[0, i] < 0.3:
        #         vel[0, i] = 0

        for i in range(len(self.vel_init)):
            self.vel_init[i] = vel[0, i]

        return vel_raw, vel

    def compute_filtered_pos(self, vel, dt):
        # integrate velocity
        pos = self.integrate(vel.copy(), self.pos_init, dt)
        pos_raw = pos.copy()
        self.raw_pos_vals = np.concatenate((self.raw_pos_vals, pos_raw))

        # band pass filter for position
        pos_bbpf = self.raw_pos_vals.copy()
        filtered_pos_x = signal.filtfilt(self.pos_b_x, self.pos_a_x, pos_bbpf[:, 0], axis=0, method="gust")
        filtered_pos_y = signal.filtfilt(self.pos_b_y, self.pos_a_y, pos_bbpf[:, 1], axis=0, method="gust")
        filtered_pos_z = signal.filtfilt(self.pos_b_z, self.pos_a_z, pos_bbpf[:, 2], axis=0, method="gust")

        len_filtered_pos = len(filtered_pos_x) - 1
        pos[0, 0] = filtered_pos_x[len_filtered_pos]
        pos[0, 1] = filtered_pos_y[len_filtered_pos]
        pos[0, 2] = filtered_pos_z[len_filtered_pos]

        for i in range(len(self.pos_init)):
            self.pos_init[i] = pos[0, i]

        return pos_raw, pos

    def display_update(self, acc_data, earth_acc_raw, filtered_earth_acc, vel_raw, filtered_vel, pos_raw, filtered_pos):
        # Data for display
        self.display = self.display.append({'LINEAR ACC X': acc_data.iloc[0, 0],
                                            'LINEAR ACC Y': acc_data.iloc[0, 1],
                                            'LINEAR ACC Z': acc_data.iloc[0, 2],
                                            'EARTH LINEAR ACC X': earth_acc_raw[0, 0],
                                            'EARTH LINEAR ACC Y': earth_acc_raw[0, 1],
                                            'EARTH LINEAR ACC Z': earth_acc_raw[0, 2],
                                            'FILTERED EARTH LINEAR ACC X': filtered_earth_acc[0, 0],
                                            'FILTERED EARTH LINEAR ACC Y': filtered_earth_acc[0, 1],
                                            'FILTERED EARTH LINEAR ACC Z': filtered_earth_acc[0, 2],
                                            'VEL X': vel_raw[0, 0],
                                            'VEL Y': vel_raw[0, 1],
                                            'VEL Z': vel_raw[0, 2],
                                            'FILTERED VEL X': filtered_vel[0, 0],
                                            'FILTERED VEL Y': filtered_vel[0, 1],
                                            'FILTERED VEL Z': filtered_vel[0, 2],
                                            'POS X': pos_raw[0, 0],
                                            'POS Y': pos_raw[0, 1],
                                            'POS Z': pos_raw[0, 2],
                                            'FILTERED POS X': filtered_pos[0, 0],
                                            'FILTERED POS Y': filtered_pos[0, 1],
                                            'FILTERED POS Z': filtered_pos[0, 2]}, ignore_index=True)

    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        high = (cutoff) / nyq
        b, a = signal.butter(order, high, btype='high', analog=False)

        return a, b

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