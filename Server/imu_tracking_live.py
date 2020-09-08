import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib; matplotlib.use("TkAgg")
from mpl_toolkits import mplot3d
import threading
import matplotlib.path as mpath
from mpl_toolkits.mplot3d import proj3d
import sys
import datetime

from position import Position

x_vals, y_vals, z_vals = [], [], []
acc_x_in, acc_y_in, acc_z_in = 0, 0, 0

fig = None
ax = None
display_gbl = pd.DataFrame()
time_gbl = 0

star = mpath.Path.unit_regular_star(3)
unit = mpath.Path.unit_regular_polygon(9)
verts = np.concatenate([unit.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([unit.codes, star.codes])
marker_style = mpath.Path(verts, codes)


class IMUTrackingLive:

    def __init__(self):

        self.anim = None
        self.trans_anim = None

        self.position = Position()

        anim_thread = threading.Thread(target=self.run_anim)
        anim_thread.start()

    def visualize(self, data, data_time):
        global acc_x_in, acc_y_in, acc_z_in, x_vals, y_vals, z_vals, display_gbl, time_gbl

        # Show current linear acceleration in the 3D plot
        acc_x_in, acc_y_in, acc_z_in = data.iloc[0, 0], data.iloc[0, 1], data.iloc[0, 2]

        # Update new linear acceleration and calculate
        pos_current, display_gbl, dt = self.position.update(data, data_time)

        x_vals.append(pos_current[0, 0])
        y_vals.append(pos_current[0, 1])
        z_vals.append(pos_current[0, 2])

        time_gbl = time_gbl + dt

    def animate_pos(self):
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

        elapsed_time = str(datetime.timedelta(seconds=time_gbl))
        fig.suptitle('3D Trajectory - ET : ' + elapsed_time, fontsize=30)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)

    def animate_filter(self):
        global fig, ax, display_gbl, time_gbl

        if len(display_gbl) > 0:
            cols_raw = ['EARTH LINEAR ACC X', 'VEL X', 'POS X',
                        'EARTH LINEAR ACC Y', 'VEL Y', 'POS Y',
                        'EARTH LINEAR ACC Z', 'VEL Z', 'POS Z']
            cols_filt = ['FILTERED EARTH LINEAR ACC X', 'FILTERED VEL X', 'FILTERED POS X',
                         'FILTERED EARTH LINEAR ACC Y', 'FILTERED VEL Y', 'FILTERED POS Y',
                         'FILTERED EARTH LINEAR ACC Z', 'FILTERED VEL Z', 'FILTERED POS Z']

            print(display_gbl.iloc[len(display_gbl) - 1, :])

            plt.cla()

            p1 = display_gbl.plot(y=cols_raw, subplots=True, sharex=True, layout=(3, 3),
                                    title=cols_raw,
                                    style='k', alpha=0.5, ax=ax)
            display_gbl.plot(y=cols_filt, subplots=True, ax=p1, style='g')

            elapsed_time = str(datetime.timedelta(seconds=time_gbl))
            fig.suptitle('Filtered Data Analysis - ET : ' + elapsed_time, fontsize=25)
            fig.tight_layout()
            plt.subplots_adjust(top=0.85)

    def animate_earth_transform(self):
        global fig, ax, display_gbl, time_gbl

        if len(display_gbl) > 0:
            cols_raw = ['LINEAR ACC X', 'LINEAR ACC Y', 'LINEAR ACC Z']
            cols_filt = ['EARTH LINEAR ACC X', 'EARTH LINEAR ACC Y', 'EARTH LINEAR ACC Z']

            print(display_gbl.iloc[len(display_gbl) - 1, :])

            plt.cla()

            p1 = display_gbl.plot(y=cols_raw, subplots=True, sharex=True, layout=(3, 1),
                                    title=cols_raw,
                                    style='k', alpha=0.5, ax=ax)
            display_gbl.plot(y=cols_filt, subplots=True, ax=p1, style='g')

            elapsed_time = str(datetime.timedelta(seconds=time_gbl))
            fig.suptitle('Earth Frame Accelerations - ET : ' + elapsed_time, fontsize=25)
            fig.tight_layout()
            plt.subplots_adjust(top=0.85)

    def run_anim(self):
        global fig, ax

        fig, ax = plt.subplots(figsize=(15, 10))
        # self.anim = FuncAnimation(fig, IMUTrackingLive.animate_pos, interval=200)
        self.anim = FuncAnimation(fig, IMUTrackingLive.animate_filter, interval=200)
        # self.anim = FuncAnimation(fig, IMUTrackingLive.animate_earth_transform, interval=200)
        plt.show()

    def stop_anim(self):
        global x_vals, y_vals, z_vals
        # x_vals.clear()
        # y_vals.clear()
        # z_vals.clear()

        self.anim.event_source.stop()
        # plt.close()



