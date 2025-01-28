import os
import re
import time

SIMU_RESULT_PATH = '../../results/'

COLLISION_TEPMLATE = ("collision detected, in epoch {time}, number of collisions: {collision_number}, number of debris generated: {debris_number}, number of debris fall down: {falldn_number}\n")
COLLISION_PATTERN = r"collision detected, in epoch (\d+\.\d+), number of collisions: (\d+), number of debris generated: (\d+)\n"

def write(epoch, timestamp, collision_number, debris_number, falldn_number, filepath, prefix="NOLAUNCH"):
    # Get the current time
    # current_time = time.strftime("%Y%m%d-%H%M%S")
    
    filename = f"{prefix}_Collision_{timestamp}.log"
    full_path = os.path.join(filepath, filename)
    
    content = COLLISION_TEPMLATE.format(time=epoch, collision_number=collision_number, debris_number=debris_number, falldn_number=falldn_number)
    
    # make sure the file exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

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

# Example Usage:
# write(1, 0.56, 1.23, 5, 'data', 0.5, 0.7)
# arrival_times, waiting_times = read(5, 'data', 0.5, 0.7)
# print(arrival_times, waiting_times)
