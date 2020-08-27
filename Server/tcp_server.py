import socket
import threading
import sys
import pandas as pd
from imu_tracking_live import IMUTrackingLive
from imu_tracking import IMUTracking


class TCPServer:

    def __init__(self, server_address, live):
        self.server_address = server_address
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.bind(self.server_address)
        except socket.error as err:
            print("Socket error: {0}".format(err))
            sys.exit(1)

        self.live = live

    def start(self):
        self.s.listen()
        print("[Listening] Server is listening on %s port %s" % self.server_address)

        while True:
            client_socket, client_address = self.s.accept()
            thread = threading.Thread(target=self.client_handle, args=(client_socket, client_address))
            thread.start()
            print(f"[Active connection] {threading.activeCount() - 1}")

    def client_handle(self, client_socket, client_address):
        print(f"\n[Client] {client_address} connected.")

        global data
        msg_length = 370
        msg_format = 'utf-8'
        msg_disconnect = "!disconnect"
        connected = True

        data = pd.DataFrame()

        if self.live:
            imu_tracking_live = IMUTrackingLive()
        else:
            imu_tracking = IMUTracking(data, 0.2)

        while connected:
            msg = client_socket.recv(msg_length).decode(msg_format)
            if msg == msg_disconnect:
                connected = False
                print(f"[{client_address}] {msg}")
                if self.live:
                    imu_tracking_live.stop_anim()
                else:
                    imu_tracking.visualize()
            else:
                if msg :
                    df = pd.DataFrame([msg.split(",")])

                    if len(df.columns) == 7:
                        if self.live:
                            data_time = int(df.iloc[0, 6])
                            imu_tracking_live.visualize(df.astype(float), data_time)
                        else:
                            imu_tracking.data_add(df.astype(float))
                    else:
                        print("Unknown data")

        client_socket.close()
        print(f"[Client] {client_address} disconnected.")








