import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap

# format: Epoch = 0, ER Nodes = 1000, Initial probability = 0.001, GC Size = 1, Avg Degree = 0.00
DYNAMIC_NETWORK_PATTERN = r"Epoch = (\d+), ER Nodes = (\d+), Initial probability = ([\d\.e\-]+), GC Size = (\d+), Avg Degree = ([\d\.]+)"
DEBRIS_REMOVAL_PATTERN = r"Epoch = (\d+), ER Nodes = (\d+), removal debris = ([\d\.e\-]+), Initial probability = ([\d\.e\-]+), GC Size = (\d+), Avg Degree = ([\d\.]+), removal rate = ([\d\.e\-]+)"

# read log files and parse the data
# TODO: combine this in the file system module
def parse_log_files(folder_path, file_prefix="dynamic_dynamic_"):
    data_dict = {}
    
    pattern = None
    if file_prefix == "dynamic_dynamic_":
        pattern = DYNAMIC_NETWORK_PATTERN
    elif file_prefix == "removal_debris_":
        pattern = DEBRIS_REMOVAL_PATTERN

    assert pattern is not None, "Invalid file prefix"

    for filename in os.listdir(folder_path):
        if filename.startswith(file_prefix):
            initial_probability_tag = None
            removal_rate_tag = None

            assert filename.endswith(".log"), "Invalid file format"
            
            file_path = os.path.join(folder_path, filename)
            file_data = []  # store the data in the file list
            with open(file_path, 'r') as file:
                for line in file:
                    match = re.match(pattern, line)
                    if match and file_prefix == "dynamic_dynamic_":
                        epoch = int(match.group(1))
                        er_nodes = int(match.group(2))
                        initial_probability = float(match.group(3))
                        initial_probability_tag = initial_probability
                        gc_size = int(match.group(4))

                        # GC fraction = GC Size / ER Nodes
                        gc_ratio = gc_size / er_nodes

                        # add the data to the list
                        file_data.append((initial_probability, epoch, gc_ratio, er_nodes))
                    
                    if match and file_prefix == "removal_debris_":
                        epoch = int(match.group(1))
                        er_nodes = int(match.group(2))
                        removal_nums = float(match.group(3))
                        gc_size = int(match.group(5))
                        removal_rate = float(match.group(7))
                        removal_rate_tag = removal_rate

                        # GC fraction = GC Size / ER Nodes
                        gc_ratio = gc_size / er_nodes

                        # add the data to the list
                        file_data.append((epoch, gc_ratio, er_nodes, removal_nums, removal_rate))
            
            if file_prefix == "dynamic_dynamic_":
                data_dict[initial_probability_tag] = file_data
            elif file_prefix == "removal_debris_":
                data_dict[removal_rate_tag] = file_data

    return data_dict

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

def plot_3d_removal_network(data_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # get the colors
    cmap = get_cmap('viridis') 
    num_colors = len(data_dict)
    colors = [cmap(i / num_colors) for i in range(num_colors)]


    # sort the data by removal rate
    # for removal_rate, data in data_dict.items():
    for (removal_rate, data), color in zip(data_dict.items(), colors):
        epochs = [item[0] for item in data]
        gc_ratios = [item[1] for item in data]
        nodes_number = [item[2] for item in data]

        # # scatter plot
        # scatter = ax.scatter(nodes_number, epochs, gc_ratios, c=gc_ratios, cmap='viridis', marker='o', label=f'Removal Rate {removal_rate:.2f}')
        
        ax.plot(nodes_number, epochs, gc_ratios, color=color, label=f'Removal Rate {removal_rate:.2f}', linewidth=2)


    # set labels
    ax.set_xlabel('ER Nodes')
    ax.set_ylabel('Epoch')
    ax.set_zlabel('GC Size / ER Nodes')
    
    # add color bar
    # fig.colorbar(scatter, ax=ax, label='GC Size / ER Nodes')


    # add legend
    ax.legend()
    plt.show()


if __name__ == "__main__":
    folder_path = "../../results"
    # data = parse_log_files(folder_path)
    # # sort the data by initial probability
    # data.sort(key=lambda x: x[0])
    # plot_3d_graph(data)

    data = parse_log_files(folder_path, "removal_debris_")
    plot_3d_removal_network(data)