import os
import re
import time

SIMU_RESULT_PATH = '../../results/'

COLLISION_TEPMLATE = ("collision detected, in epoch {time}, number of collisions: {collision_number}, number of debris generated: {debris_number}, number of debris fall down: {falldn_number}\n")
COLLISION_PATTERN = r"collision detected, in epoch (\d+\.\d+), number of collisions: (\d+), number of debris generated: (\d+)\n"

STATIC_NETWORK_TEMPLATE = ("ER Nodes = {N}, Initial probability = {probability}, GC Size = {gc_size}, Avg Degree = {avg_degree:.2f}\n")
STATIC_NETWORK_PATTERN = r"ER Nodes = (\d+), Initial probability = (\d+\.\d+), GC Size = (\d+), Avg Degree = (\d+\.\d+)\n"

DYNAMIC_NETWORK_TEMPLATE = ("Epoch = {iteration}, ER Nodes = {N}, Initial probability = {probability}, GC Size = {gc_size}, Avg Degree = {avg_degree:.2f}\n")
DYNAMIC_NETWORK_PATTERN = r"Epoch = (\d+), ER Nodes = (\d+), GC Size = (\d+), Avg Degree = (\d+\.\d+)\n"

REMOVAL_NETWORK_TEMPLATE = ("Epoch = {iteration}, ER Nodes = {N}, removal debris = {rate}, Initial probability = {probability}, GC Size = {gc_size}, Avg Degree = {avg_degree:.2f}, removal rate = {rate}\n")
REMOVAL_NETWORK_PATTERN = r"Epoch = (\d+), ER Nodes = (\d+), removal debris = (\d+\.\d+), Initial probability = (\d+\.\d+), GC Size = (\d+), Avg Degree = (\d+\.\d+), removal rate = (\d+\.\d+)\n"

def write(epoch, timestamp, collision_number, debris_number, falldn_number, N, removal_rate, probability, gc_size, avg_degree, filepath, prefix="NOLAUNCH", model="static"):
    switcher = {
        "static": STATIC_NETWORK_TEMPLATE,
        "dynamic": DYNAMIC_NETWORK_TEMPLATE,
        "collision": COLLISION_TEPMLATE,
        "removal": REMOVAL_NETWORK_TEMPLATE
    }
    template = switcher.get(model, "Invalid model")
    assert template != "Invalid model", "Invalid model"

    filename = f"{model}_{prefix}_{timestamp}.log"
    # make sure the file exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    full_path = os.path.join(filepath, filename)
    
    content = ""
    if model == "collision":
        content = template.format(time=epoch, collision_number=collision_number, debris_number=debris_number, falldn_number=falldn_number)
    elif model == "static":
        content = template.format(N=N, probability=probability, gc_size=gc_size, avg_degree=avg_degree)
    elif model == "dynamic":
        content = template.format(iteration=epoch, N=N, probability=probability, gc_size=gc_size, avg_degree=avg_degree)
    elif model == "removal":
        content = template.format(iteration=epoch, N=N, rate=removal_rate, probability=probability, gc_size=gc_size, avg_degree=avg_degree)
        
    with open(full_path, 'a') as file:
        file.write(content)

def read(prefix="NOLAUNCH"):
    results = list()

    file_prefix = f"{prefix}_Collision_"  # The prefix of the file
    current_dir = os.getcwd()  # The current directory
    files = [f for f in os.listdir(current_dir) if f.startswith(file_prefix)]

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Reading {file}:\n{content}\n")
            
            # Parse the content
            matches = re.findall(COLLISION_PATTERN, content)
            if len(matches) == 0:
                return None, None
            else:
                results.append(matches)

    return results

def check_directory(directory):
    return os.path.exists(directory)

def create_directory(directory):
    os.makedirs(directory)
