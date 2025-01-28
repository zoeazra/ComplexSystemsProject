import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# random 3D positions
N = 1000
positions = np.random.rand(N, 3) * 1000

# randomly select 200 positions as GC
gc_indices = np.random.choice(N, size=200, replace=False)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5, alpha=0.5, label="Non-GC")
ax.scatter(positions[gc_indices, 0], positions[gc_indices, 1], positions[gc_indices, 2],
           s=20, color='red', label="GC")
plt.legend()
plt.title("3D Distribution of Orbital Fragments")
plt.show()
