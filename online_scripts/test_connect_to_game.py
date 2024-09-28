from UDPHandler import GameUDPInterface

udp_handler = GameUDPInterface(verbose = True)

udp_handler.connect()

while True:
    try:
        pass
    # Example: Sending a message to a remote server
    # udp_handler.send_message('\00\00\01\01\01\01\01\01\01\02', "127.0.0.1", 59075)

    # Example: Receiving a message
    # message, address = udp_handler.receive_message()
    # print(message)
    # print(address)

    except KeyboardInterrupt:
        udp_handler.disconnect()