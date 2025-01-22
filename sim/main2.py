import sys
import numpy as np
from tqdm import tqdm
import csv
import random
import matplotlib.pyplot as plt

from model import *
from view import View
from data_cleaning import data_array, all_groups

def fast_arr(objects: np.ndarray):
    """
    Prepare fast array for usage with Numba.

    Returns array of the form:
      -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL'  'pos_x', pos_y', 'pos_z']
    """
    return np.array(
        [[object[0], object[4], object[6], object[13], 0, 0, 0] for object in objects]
    )

def plot_graphs(debris_file: str, collisions_file: str):
    """
    Plot the cumulative number of debris and collisions over time based on the data in the CSV files.

    debris_file: Path to the CSV file containing debris data.
    collisions_file: Path to the CSV file containing collisions data.
    """
    # Debris Data
    times_debris = []
    debris_counts = []
    cumulative_debris = 0
    with open(debris_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            debris_added = int(row[0])
            cumulative_debris += debris_added
            times_debris.append(float(row[1]))
            debris_counts.append(cumulative_debris)

    # Collisions Data
    times_collisions = []
    collision_counts = []
    collision_time_map = {}

    with open(collisions_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            time = float(row[2])
            if time in collision_time_map:
                collision_time_map[time] += 1
            else:
                collision_time_map[time] = 1

    # Sort and prepare cumulative collision data
    sorted_times = sorted(collision_time_map.keys())
    cumulative_collisions = 0
    for time in sorted_times:
        cumulative_collisions += collision_time_map[time]
        times_collisions.append(time)
        collision_counts.append(cumulative_collisions)

    # Generate the graphs
    plt.figure(figsize=(12, 6))

    # Plot Debris
    plt.subplot(1, 2, 1)
    plt.plot(times_debris, debris_counts, marker='o', label="Cumulative Debris Count", color='blue')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cumulative Number of Debris")
    plt.title("Debris Over Time")
    plt.legend()
    plt.grid(True)

    # Plot Collisions
    plt.subplot(1, 2, 2)
    plt.plot(times_collisions, collision_counts, marker='o', color='orange', label="Cumulative Collisions")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Cumulative Number of Collisions")
    plt.title("Collisions Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def run_sim(
    objects: np.ndarray,
    group: int,
    draw: bool,
    margin: float,
    endtime: float,
    timestep: float,
    epoch: float,
    probability: float,
    percentage: float,
    frequency_new_debris: int,
) -> tuple[list, list, list]:
    """
    Run the simulation by calculating the position of the objects, checking
    for collisions and handling the collisions.

    objects: array of objects in the following form:
    -> ['EPOCH', 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER',
       'MEAN_ANOMALY', 'NORAD_CAT_ID', 'SEMIMAJOR_AXIS', 'OBJECT_TYPE',
       'RCS_SIZE', 'LAUNCH_DATE', 'positions', 'rotation_matrix', 'groups'].
    group: number of the orbit group.
    draw: if true an animation will be started in the browser.
    margin: threshold of when two objects are colliding.
    endtime: end time of the simulation in seconds.
    timestep: size of the time steps in seconds.
    epoch: Julian date in seconds of the start of the simulation.
    probability: The probablity of adding new debris per call of the function
    "random_debris".
    percentage: percentage of the number of existing debris to add every call
    of "random_debris".
    frequency_new_debris: frequency of calling the function "random_debris".
    If this value is 100 and the timestep is 100, "random_debris" will be called
    every 100x100 seconds.

    Returns a tuple of the simulation parameters, new debris and collision data.
    """

    if draw:
        view = View(objects)

    initialize_positions(objects, epoch)
    objects_fast = fast_arr(objects)
    matrices = np.array([object[11] for object in objects])

    parameters, collisions, added_debris = [], [], []

    for time in tqdm(
        range(int(epoch), int(epoch + endtime), timestep),
        ncols=100,
        desc=f"group: {group}",  # tqdm for the progress bar.
    ):
        calc_all_positions(objects_fast, matrices, time)

        if len(objects_fast) > 2000:
            print(f"\nGroup {group} process killed.")
            sys.exit()

        collided_objects = check_collisions(objects_fast, margin)
        if collided_objects != None:

            object1, object2 = collided_objects[0], collided_objects[1]
            # Compute new debris
            new_debris = collision(object1, object2)

            # Add new debris to the total objects array and a random matrix to
            # the matrices array.
            objects_fast = np.concatenate((objects_fast, new_debris), axis=0)
            new_matrix = matrices[random.randint(0, len(matrices) - 1)]
            matrices = np.concatenate((matrices, [new_matrix]), axis=0)

            # Save the collision data
            collisions.append([object1, object2, time])

        if (
            frequency_new_debris != None
            and (time - epoch) % (frequency_new_debris * timestep) == 0
        ):  # Add new debris at timesteps indicated by frequency_new_debris.
            objects_fast, matrices, new_debris = random_debris(
                objects_fast, matrices, time, percentage
            )
            added_debris.append([new_debris, time])

            if draw:
                view.make_new_drawables(objects_fast)

        if draw:
            view.draw(objects_fast, time - epoch)

    parameters.append(
        [objects[0][12], epoch, endtime, timestep, probability, percentage]
    )

    return parameters, collisions, added_debris


if __name__ == "__main__":

    if len(sys.argv) > 1 and int(sys.argv[1]) in all_groups:
        group = int(sys.argv[1])

    else:
        print("\nGive a valid number of the orbit you want to evaluate")
        sys.exit()

    # select given group.
    group_selection = data_array[:, 12] == group
    data_array_group = data_array[group_selection]
    objects = data_array_group

    # Activate / don't activate the view.
    draw = False
    if len(sys.argv) > 2 and sys.argv[2] == "view":
        draw = True

    parameters, collisions, added_debris = run_sim(
        objects,
        group,
        draw,
        margin=70000,
        endtime=1_000_000,
        timestep=5,
        epoch=1675209600.0,
        probability=0,
        percentage=0,
        frequency_new_debris=None,
    )

    # Save the data to "sim_data" in the correct group folder.

    debris_file = f"sim_data/group_{objects[0][12]}/debris.csv"
    collisions_file = f"sim_data/group_{objects[0][12]}/collisions.csv"

    with open(f"sim_data/group_{objects[0][12]}/parameters.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(
            ["group", "epoch", "endtime", "timestep", "probabilty", "precentage"]
        )
        write.writerows(parameters)

    with open(collisions_file, "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["object1", "object2", "time"])
        write.writerows(collisions)

    with open(debris_file, "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["number_debris", "time"])
        write.writerows(added_debris)

    print(f"\ngroup {group} done running")

    # Generate graphs if the simulation is not in 'draw' mode
    if not draw:
        print("\nGenerating debris and collision graphs...")
        plot_graphs(debris_file, collisions_file)
