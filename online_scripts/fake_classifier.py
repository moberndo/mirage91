from pylsl import StreamInfo, StreamOutlet
import time
from pynput import keyboard

import logging

# Suppress LSL-related log output
logging.getLogger('pylsl').setLevel(logging.ERROR)


# Create LSL stream
info = StreamInfo(name='KeyPressStream', type='ClassProb', nominal_srate=10, channel_count=4, channel_format='float32', source_id='uniqueid1234')
outlet = StreamOutlet(info)

print("Press 1, 2, 3, or 4 to send data to the LSL stream. Press 'q' to quit.")

# Frequency of 10Hz means a sample is sent every 0.1 seconds
frequency = 10  # Hz
interval = 1.0 / frequency  # time interval between samples (0.1 seconds for 10Hz)

value_to_send = [.25, .25, .25, .25]

# Function to handle key presses
def on_press(key):
    global value_to_send

    try:
        if key.char == '1':
            print("Button 1 pressed")
            value_to_send = [0.7, 0.1, 0.1, 0.1]  # Set value for key 1
        elif key.char == '2':
            print("Button 2 pressed")
            value_to_send = [0.1, 0.7, 0.1, 0.1]  # Set value for key 2
        elif key.char == '3':
            print("Button 3 pressed")
            value_to_send = [0.1, 0.1, 0.7, 0.1]  # Set value for key 3
        elif key.char == '4':
            print("Button 4 pressed")
            value_to_send = [0.1, 0.1, 0.1, 0.7]  # Set value for key 4
        elif key.char == 'q':
            print("Exiting...")
            return False  # Stop listener, exits the script
    except AttributeError:
        pass  # Handle special keys

# Start listening to keyboard events
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while True:
        start_time = time.time()

        # Send the value to the LSL stream
        outlet.push_sample(value_to_send)

        # Ensure we are sending data at 10Hz (every 0.1 seconds)
        elapsed_time = time.time() - start_time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

except KeyboardInterrupt:
    print("Stopped manually.")
finally:
    listener.stop()
