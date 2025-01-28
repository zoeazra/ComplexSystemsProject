import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# format: Epoch = 0, ER Nodes = 1000, Initial probability = 0.001, GC Size = 1, Avg Degree = 0.00
DYNAMIC_NETWORK_PATTERN = r"Epoch = (\d+), ER Nodes = (\d+), Initial probability = ([\d\.e\-]+), GC Size = (\d+), Avg Degree = ([\d\.]+)"

# read log files and parse the data
# TODO: combine this in the file system module
def parse_log_files(folder_path, file_prefix="dynamic_dynamic_"):
    data = []

    for filename in os.listdir(folder_path):
        if filename.startswith(file_prefix):
            assert filename.endswith(".log"), "Invalid file format"

            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    match = re.match(DYNAMIC_NETWORK_PATTERN, line)
                    if match:
                        epoch = int(match.group(1))
                        er_nodes = int(match.group(2))
                        initial_probability = float(match.group(3))
                        gc_size = int(match.group(4))

                        # GC fraction = GC Size / ER Nodes
                        gc_ratio = gc_size / er_nodes

                        # add the data to the list
                        data.append((initial_probability, epoch, gc_ratio))

    return data

# 3d plot of the data
def plot_3d_graph(data):
    # extract the data
    initial_probabilities = [item[0] for item in data]
    epochs = [item[1] for item in data]
    gc_ratios = [item[2] for item in data]

    max_initial_probability = max(initial_probabilities)
    min_initial_probability = min(initial_probabilities)
    adjusted_initial_probabilities = [max_initial_probability - p for p in initial_probabilities]

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # scatter plot
    scatter = ax.scatter(adjusted_initial_probabilities, epochs, gc_ratios, c=gc_ratios, cmap='viridis', marker='o')

    # 调整 X 轴刻度
    x_ticks = [max_initial_probability - p for p in np.linspace(min_initial_probability, max_initial_probability, num=5)]
    x_tick_labels = [f"{p:.2e}" for p in np.linspace(min_initial_probability, max_initial_probability, num=5)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    # set labels
    ax.set_xlabel('Initial Probability')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('GC Size / ER Nodes')

    # add color bar
    fig.colorbar(scatter, ax=ax, label='GC Size / ER Nodes')

    # show the plot
    plt.show()

if __name__ == "__main__":
    folder_path = "../../results"
    data = parse_log_files(folder_path)

    # sort the data by initial probability
    data.sort(key=lambda x: x[0])
    plot_3d_graph(data)
