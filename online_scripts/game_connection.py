import socket
from pylsl import StreamInlet, resolve_stream

def send_connection_request():
    # Send a connection request to the game
    request_message = b'connect_request'  # Adjust based on manual
    sock.sendto(request_message, (UDP_IP, UDP_PORT))

def receive_connection_token():
    # Wait for the game to send the connection token
    data, addr = sock.recvfrom(1024)  # Buffer size 1024 bytes
    connection_token = data  # Extract the connection token
    print(f"Received connection token: {connection_token}")
    return connection_token

# Example of sending a packet with a header and some data
def send_bci_data(data):
    # Create the header (10 bytes) and payload (variable length)
    header = create_packet_header()
    payload = create_payload(data)
    
    # Combine header and payload
    message = header + payload
    
    # Send the UDP packet
    sock.sendto(message, (UDP_IP, UDP_PORT))

def create_packet_header():
    # Example of creating a 10-byte header
    header = b'\x00' * 10  # adjust based on the manual's structure
    return header

def create_payload(data):
    # Convert the EEG signal or command into bytes
    payload = data.encode()  # placeholder, format accordingly
    return payload




# Find the stream
streams = resolve_stream('name', 'ClassifierOutput')
inlet = StreamInlet(streams[0])

# Receive data from the stream
while True:
    sample, timestamp = inlet.pull_sample()
    print(f"Received: {sample} at {timestamp}")


'''# Set up the socket for communication
UDP_IP = "your_game_host_ip"
UDP_PORT = 59075  # as specified for BCI input stream

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# Example of sending a command
send_bci_data("your_bci_control_signal")'''
