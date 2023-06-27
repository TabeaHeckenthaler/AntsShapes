import keyboard
import time
import matplotlib.pyplot as plt

# Create a list to store the timestamps of space bar presses
space_bar_presses = []

# Function to handle space bar press event
def on_space(event):
    if event.name == 'space':
        space_bar_presses.append(time.time())

# Register the space bar press event handler
keyboard.on_press(on_space)

# Keep the program running until interrupted
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Print the recorded timestamps
    print("Space bar presses:")
    for timestamp in space_bar_presses:
        print(time.ctime(timestamp))

    # Plot the results
    num_presses = len(space_bar_presses)
    x = range(num_presses)
    y = space_bar_presses

    plt.plot(x, y, 'bo')
    plt.xlabel('Press Number')
    plt.ylabel('Timestamp')
    plt.title('Space Bar Presses Over Time')
    plt.show()
