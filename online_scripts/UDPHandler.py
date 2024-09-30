import socket
import secrets
import threading
import time
import struct

# ---Header Composition--- (10 Bytes)

# Type of the package (Byte 0)
# 00 Connection request (send to game)
# 01 Connection reject (receive from game)
# 02 Connection accept (receive from game)
# 03 Disconnect (receive from game)
# 04 Data (send to game)
# 05 Ping (send and receive)
# 06 Pong (send to game)

# Flag (Byte 1)
# 0000 000_ Has connection Token
# 0000 00_x Has pipeline (not sure what this is)

# Session token (Byte 2-9)
# Generally the session token in the header is the receiver.
# Only during the connection request that we have to send the client token instead

# ---Payload Composition--- (4 Bytes)

# Binary Inputs (Byte 10-11)
# 000 000_ Input A
# 000 00_x Input B

# Analogue Inputes (Byte 12-13)
# xx yy (range from -1 to 1 or in Hex 00h-FFh)

# ---Connection Token--- (8 Bytes; only when first request connection)
# Refered to as Host token which we have to address when sending commend to host

#local_ip='169.254.105.17', local_port=59075
class GameUDPInterface:
    def __init__(self, host_ip = '127.0.0.1', host_port = 59075, local_ip='0.0.0.0', local_port=64385, buffer_size=1024, verbose = False):
        """
        Initializes the UDPHandler for reading and writing.
        
        :param local_ip: IP address to bind the socket to (default is '0.0.0.0' to accept all IPs).
        :param local_port: Port to bind the socket to.
        :param buffer_size: Size of the buffer for reading messages.
        """
        self.hearbeat_interval = 25
        self.verbose = verbose
        self.host_ip = host_ip
        self.host_port = host_port
        self.local_ip = local_ip
        self.local_port = local_port
        self.buffer_size = buffer_size

        self.client_token = self._create_token()
        self.host_token = ''

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind the socket to an IP and port for reading
        self.sock.bind((self.local_ip, self.local_port))
        # self.sock.settimeout(5) 
        if self.verbose:
            print(f"UDP Socket bound to {self.local_ip}:{self.local_port}")

        self.connected = False

        self.heartbeat_thread = threading.Thread(target=self._send_heartbeat)
        self.heartbeat_thread.daemon = True

    def create_packet_header(self, packet_type, session_token, has_token=False):
        # packet header based on game protocol p.64
        flags = 0x01 if has_token else 0x00
        header = (
            packet_type.to_bytes(1, byteorder='big') +
            flags.to_bytes(1, byteorder='big') +
            session_token
        )
        return header

    # def create_payload(self, data):
    #     # Convert the EEG signal into bytes, --> float (32-bit) to bytes, p.71
    #     payload = bytearray()
    #     for value in data:
    #         payload.extend(struct.pack('<f', float(value)))  # <f means little-endian float
    #     return payload
    

    def _create_token(self,token_size=8):
        return secrets.token_hex(token_size)

    # def get_client_token(self):
    #     return self.client_token

    # def get_host_token(self):
    #     return self.host_token

    def connect(self):
        """
        Establish initial handshake
        """
        self._send_message('0000'+self.client_token)
        type_resp, flag_resp, session_token, connection_token = self._receive_message()

        if type_resp == '01': 
            if self.verbose:
                print('Connection Rejected')
        if type_resp == '02':
            if self.verbose:
                print('Connection Accepted')
            # print(bytes.fromhex(flag_resp))
            # check Has connection token bit
            if bytes.fromhex(flag_resp)[0] & 0x01:
                if session_token != self.client_token:
                    print('Incorrect client_token received')

                self.host_token = connection_token
                if self.verbose:
                    print('Host token received:', self.host_token)
                self.connected = True
            self.heartbeat_thread.start()


    def _unfold_package(self, message):
        """
        Unfold and extract the incoming package
        """
        if len(message) >= 1:
            # Header size is 10 bytes or 20 in Hex
            header = message[0:20]
            type_resp = header[0:2]
            flag_resp = header[2:4]
            session_token = header[4:20]
            connection_token = []

            if type_resp == '02':
            # response size is 18 bytes. No payload but contains connection token
                connection_token = message[20:36]

        return type_resp, flag_resp, session_token, connection_token


    def _send_message(self, message):
        """
        Sends a UDP message to a remote IP and port.
        
        :param message: The message to send (as bytes).
        :param remote_ip: The IP address of the remote machine.
        :param remote_port: The port of the remote machine.
        """
        self.sock.sendto(bytes.fromhex(message), (self.host_ip, self.host_port))
        # if self.verbose:
        #     print(f"Sent message: '{message}' to {self.host_ip}:{self.host_port}")

    def _receive_message(self):
        """
        Receives a UDP message from any sender.
        
        :return: A tuple containing the message and the address (ip, port) of the sender.
        """
        # while True:
        try:
            data, addr = self.sock.recvfrom(self.buffer_size)
        except KeyboardInterrupt:
            pass
        return self._unfold_package(bytes.hex(data))

    def _send_heartbeat(self):
        """
        Regularly send heartbeart to keep connection every 2 sec. (according to the manual)
        """
        while True:
            if self.connected:
                # the token in the header is the receiver
                self._send_message('0500'+self.host_token)
                if self.verbose:
                    print('Ping')

                type_resp, flag_resp, session_token, connection_token = self._receive_message()
                if type_resp == '06':
                    # The received token in the header should be client token
                    if self.verbose:
                        print('Pong received')
                        if session_token == self.client_token:
                            print('Correct client token received')
                        else:
                            print('Incorrect client token received')
                elif type_resp == '05': 
                    # The host is pinging the client, so we should pong back with the host token
                    self._send_message('0600'+self.host_token)
                    if self.verbose:
                        print('Pong')
            time.sleep(self.hearbeat_interval)

    def disconnect(self):
        """Closes the UDP socket."""
        self._send_message('0300'+self.host_token)
        self.sock.close()
        if self.verbose:
            print("Socket closed")
