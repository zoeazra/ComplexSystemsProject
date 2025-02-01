import matplotlib.pyplot as plt
import numpy as np

# Defining the x array with more points for smoother transitions
x = np.linspace(0, 4, 1200)  # More points for an even smoother curve

def final_wave(x):
    # Break the waveform into segments with longer flats and direct vertical drops
    y = np.zeros_like(x)
    for i in range(int(np.ceil(max(x)))):
        mask = (x >= i) & (x < i + 1)
        segment_x = x[mask] - i  # Local x for each segment
        # Defining a clear vertical drop at 0.7
        y[mask] = np.where(segment_x < 0.2, 0,
                           np.where(segment_x < 0.7, 5 * (segment_x - 0.2)**2,
                                    0))  # Drop immediately to 0 without lateral movement
    return y

# Evaluate the function
y = final_wave(x)

# Plotting the final refined function
plt.figure(figsize=(10, 5))
plt.plot(x / 4 * 100, y / max(y) * 100)  # Rescaling x and y to be between 0 and 100

# Pastel red color definition (lighter and softer)
pastel_red = (1, 0.4, 0.4)  # Adjusted RGB values for a softer red

# Adding red dashed vertical lines exactly where the rise starts, in pastel red
rise_starts = np.array([0.2, 1.2, 2.2, 3.2])  # Adding 0.2 to each starting point to shift right to the rise start
for start in rise_starts:
    plt.axvline(x=start / 4 * 100, color=pastel_red, linestyle='--')

plt.title('Potential SOC')
plt.xlabel('Time')
plt.ylabel('Collisions')
plt.grid(True)
plt.legend(['System Dynamics', 'Kessler Emergence'])
plt.show()
