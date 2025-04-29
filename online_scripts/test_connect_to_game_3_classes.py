'''# # from UDPHandler import GameUDPInterface

# # udp_handler = GameUDPInterface(verbose = True)

# # udp_handler.connect()

# # while True:
# #     try:
# #         pass
# #     # Example: Sending a message to a remote server
# #     # udp_handler.send_message('\00\00\01\01\01\01\01\01\01\02', "127.0.0.1", 59075)

# #     # Example: Receiving a message
# #     # message, address = udp_handler.receive_message()
# #     # print(message)
# #     # print(address)

# #     except KeyboardInterrupt:
# #         udp_handler.disconnect()

# from UDPHandler import GameUDPInterface
# from pylsl import StreamInlet, resolve_stream
# import struct

# #NOTE: summarized test_connect_to_game.py and game_connection.py into this file instead since it is repetative.
# def create_packet_header(packet_type, session_token, has_token=False):
#     # packet header based on game protocol p.64
#     flags = 0x01 if has_token else 0x00
#     header = (
#         packet_type.to_bytes(1, byteorder='big') +
#         flags.to_bytes(1, byteorder='big') +
#         session_token
#     )
#     return header

# def create_payload(data):
#     # convert the EEG signal into bytes, --> float (32-bit) to bytes
#     payload = bytearray()
#     for value in data:
#         payload.extend(struct.pack('<f', float(value)))  # <f means little-endian float
#     return payload

# # initialize the UDP handler
# udp_handler = GameUDPInterface(verbose=True)
# print(f"UDP Socket bound to {udp_handler.local_ip}:{udp_handler.local_port}") # log: local or host?

# udp_handler.connect() # connect to game
# print("Connection Accepted") #log

# # find the LSL stream
# try:
#     streams = resolve_stream('name', 'ClassifierOutput')
#     inlet = StreamInlet(streams[0])
#     print("LSL Stream Found")  # confirm find, Log
# except Exception as e:
#     print(f"Failed to resolve LSL stream: {e}")
#     exit()  # Exit stream, Log

# # send classification data
# while True:
#     try:
#         sample, timestamp = inlet.pull_sample()  #üull the latest sample from LSL stream
#         print(f"Received classification: {sample} at {timestamp}")  #log

#         session_token = udp_handler.client_token  # client token from handler
#         connection_token = udp_handler.host_token  ####  does it work like that?

#         header = udp_handler.create_packet_header(packet_type=0x04, session_token=session_token, has_token=True)

#         payload = udp_handler.create_payload(sample)

#         message = header + payload + connection_token #UDP message
#         udp_handler._send_message(message.hex())  # send message to the game via UDP in hex format
#         print(f"Sent BCI data to game: {message.hex()}")  #log

#     #Error functions
#     except KeyboardInterrupt:
#         print("Disconnecting...")
#         udp_handler.disconnect()
#         break
#     except Exception as e:
#         print(f"Error occurred: {e}. The game may have closed or is unreachable.")
#         udp_handler.disconnect() 
#         break'''

from numpy import ones, append, array, copy, reshape, expand_dims, zeros_like, zeros, mean


from UDPHandler import GameUDPInterface
from pylsl import StreamInlet, resolve_stream
import struct

# initialize the buffer
num_class = 3
buffer_size = 5
buffer = ones(shape=(num_class,1))

# Initialize the UDP handler
udp_handler = GameUDPInterface(host_ip='10.16.10.20', verbose=True)
print(f"UDP Socket bound to {udp_handler.local_ip}:{udp_handler.local_port}") # log: local or host?

try:
    udp_handler.connect()  # Connect to game
    print("Connection Accepted")  # Log
except Exception as e:
    print(f"Error occurred: {e}. The game may have closed or is unreachable.") # disconnection error 
    udp_handler.disconnect()
    

# Find the LSL stream
try:
    streams = resolve_stream('name', 'ClassifierOutput')
    inlet = StreamInlet(streams[0])
    print("LSL Stream Found")  # Confirm find, log
except Exception as e:
    print(f"Failed to resolve LSL stream: {e}")
    exit()  # Exit stream, log
    
    
 #################### create payload acc. to p.71 #####################
        
# classify binary input A,B
def classify_binary_inputs(sample):
    # mask for binary inputs, initialized all 16 bits to 0 for starters
    binary_inputs = 0
    # thrsh = 0.5
    # if sample[0] > sample[1]:
    #     if sample[0] > thrsh:
    #         binary_inputs |= 1<<8 # Input A (Bit 1)
    # else:
    #     if sample[1] > thrsh:
    #         binary_inputs |= 2<<8  # Input B (Bit 2)
    # # put the 16-bit binary inputs into bytes
    return binary_inputs.to_bytes(2, byteorder= 'big') # little = little-endian

# Function to encode analogue inputs into the range [00h, FFh] (1 byte per axis)
def encode_analogue_input(value,thresh,direction='positive'):
    # analogue input from [-1.0, 1.0] to [00h, FFh]
    # print(value)
    max_prob = 0.8
    if value >= thresh:
        # encoded_value = int((value + 1) * 127.5)  # [-1, 1] -> [0, 255]
        # encoded_value = int((((value -0.5)/0.25)*128)+127)
        if thresh <= value < max_prob:
            # print(value)
            if direction == 'positive':
                encoded_value = int((((value - thresh)/(max_prob-thresh))*128)+127)  # y = 0.5 for 0.5 <= x < 0.7
            elif direction == 'negative':
                encoded_value = int(((-(value - thresh)/(max_prob-thresh))*128)+127)  # y = 0.5 for 0.5 <= x < 0.7
        elif max_prob <= value <= 1:
            if direction == 'positive':
                encoded_value = int(255)
            elif direction == 'negative':
                encoded_value = int(0)
        # encoded_value = max(0, min(255, encoded_value))
    else:
        encoded_value = int(127)
    return encoded_value.to_bytes(1, byteorder= 'little')

def create_payload(sample_buffer):
    def encode_turning(value, thresh_x, min_speed=0.5, max_speed=0.9):
        x_axis = int(-((value-thresh_x) / (1 - thresh_x) * (128*max_speed)) + (127))
        return x_axis.to_bytes(1, byteorder= 'little')
    # Binary Inputs A,B (2 bytes for 16-bit mask)
    binary_inputs = classify_binary_inputs(sample_buffer)

    thresh_x = 0.75
    thresh_y = 0.85

    rest_thresh = 0.75

    index_LH = 0
    index_F = 1
    index_rest =2

    global counter

    if sample[index_rest] < rest_thresh:
        # for y axis (LH)
        if sum(sample_buffer[index_LH,:] > thresh_x) == buffer_size:
            # send x axis command
            x_axis = encode_turning(value=mean(sample_buffer[index_LH,:]), thresh_x=thresh_x)
            counter += 1
            print(counter)
            if counter >= 5:
                # set binary output a true
                binary_inputs = 0
                binary_inputs |= 1<<8 
                binary_inputs = binary_inputs.to_bytes(2, byteorder='big')
                counter = 0
        else:
            x_axis = int(127).to_bytes(1, byteorder= 'little')
            counter = 0

        if sample_buffer[index_F,-1] > thresh_y:
            # send x axis command
            y_axis = int(255*0.7).to_bytes(1, byteorder= 'little')
        else:
            y_axis = int(127).to_bytes(1, byteorder= 'little')

    else:
        x_axis = int(127).to_bytes(1, byteorder= 'little')
        y_axis = int(127).to_bytes(1, byteorder= 'little')


    
        
    # Analogue Inputs (X and Y axes) mapping
    
    if sample[2] < rest_thresh:
        # the negative parameter should flip the output in the opposite direction
        x_axis = encode_analogue_input(sample[1],thresh_x,direction = 'negative')  # 3rd value to X-axis
        y_axis = encode_analogue_input(sample[0],thresh_y)  # 4th value to Y-axis
    else:
        x_axis = int(127).to_bytes(1, byteorder= 'little')
        y_axis = int(127).to_bytes(1, byteorder= 'little')

    # print(f"{sample[2]} : {x_axis}")
    # print(f"{sample[3]} : {y_axis}")
    # Combine binary and analogue inputs into the full payload
    return binary_inputs + x_axis + y_axis

###################################
#counter for the a button during class 1        
counter = 0
# Send classification data
while True:
    try:
        sample, timestamp = inlet.pull_sample()  # Pull the latest sample from LSL stream
        sample = array(sample).reshape((3, 1))
        # print(f"Received classification: {sample} at {timestamp}")  # Log

        # session_token = udp_handler.client_token  # Client token from handler
        # connection_token = udp_handler.host_token  # Host token
        # header = udp_handler.create_packet_header(packet_type=0x04, session_token=session_token, has_token=True)
        
        buffer = append(buffer, sample, axis=1)

        if buffer.shape[1] >= buffer_size:
            buffer = buffer[:, -buffer_size:]

            session_token = bytes.fromhex(udp_handler.client_token)  # transform to bytes
            connection_token = bytes.fromhex(udp_handler.host_token)  # transform to bytes

            # Step 3: Create the packet header
            header = udp_handler.create_packet_header(packet_type=0x04, session_token=connection_token, has_token=True)

            payload = create_payload(buffer)
            # print(f'Payload: {payload}')
            message = header + payload # UDP message
            print(message.hex())
            udp_handler._send_message(message.hex())  # Send message to the game via UDP in hex format
            # print(f"Sent BCI data to game: {message.hex()}")  # Log

    # Error handling
    except KeyboardInterrupt:
        print("Disconnecting...")
        udp_handler.disconnect()
        break
    # except Exception as e:
    #     print(f"Error occurred: {e}. The game may have closed or is unreachable.")
    #     udp_handler.disconnect()
    #     break
