from tcp_server import TCPServer

port = 3000
server = "192.168.111.119"
server_address = (server, port)

print("[Starting] Server is starting...")
live = True
tcp_server = TCPServer(server_address, live)
tcp_server.start()

