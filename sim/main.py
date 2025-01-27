"""
"main.py"

Run this module to start the simulation.

Usage:
-> python main.py [orbit number (between 0 and 99)] [view (optional)]

* If "view" is given, an animation will start up in the browser.
* Data will be saved to "sim_data" in the folder corresponding to the group number.
"""


import sys
import numpy as np
from tqdm import tqdm
import csv
import random
import file_sys
import time as tf

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
       'RCS_SIZE', 'LAUNCH_DATE', 'positions', 'rotation_matrix', 'groups', 'object_bool'].
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

    # if draw:
    #     view = View(objects)

    #objects = objects[0 : INI_NUMBERS-1] # pick up the first INI_NUMBERS objects from the real data
    INI_NUMBERS = len(objects)
    INI_DEBRIS = 30
    initialize_positions(objects, epoch)
    objects_fast = fast_arr(objects)
    matrices = np.array([object[11] for object in objects])

    print(f"Group {group} process has object_fast shape: {objects_fast.shape}, matrices shape: {matrices.shape}")

    if draw:
        view = View(objects_fast)

    parameters, collisions, added_debris = [], [], []
    current_time = tf.strftime("%Y%m%d-%H%M%S")

    # initialize the debris
    for _ in range(INI_DEBRIS):
        objects_fast, matrices, new_debris_num = random_debris(
            objects_fast, matrices, epoch, percentage
        )
        added_debris.append([new_debris_num, epoch])

    post_collision = False

    for time in tqdm(
        range(int(epoch), int(epoch + endtime), timestep),
        ncols=100,
        desc=f"group: {group}",  # tqdm for the progress bar.
    ):  
    #for time in range(int(epoch), int(epoch + endtime), timestep):
        #print(f"\nGroup {group} process running at time {time}.")
        calc_all_positions(objects_fast, matrices, time)

        if len(objects_fast) > 500000000:
            print(f"\nGroup {group} process killed.")
            sys.exit()

        # issue 1, make a list of all the objects that are colliding, not just the first collision
        # issue 2, should update the matrices for the new debris
        collision_pairs = check_collisions_optimized(objects_fast, margin)
        total_debris_generated_this_epoch = 0
        falldown_number = 0

        # when the time is greater than the epoch + 0.5 * endtime, launch 10 satellites frequently
        if time > epoch +  0.5 * endtime and (time % (frequency_new_debris * timestep)) == 0:
            print(f"launching 10 satellite at time {time}")
            for _ in range(10):
                objects_fast, matrices = launch_satellites(
                    objects_fast, matrices, time
                )
        
        if len(collision_pairs) != 0:
            post_collision = True
            for collided_objects in collision_pairs:
                index1, index2, object1, object2 = collided_objects[0], collided_objects[1], collided_objects[2], collided_objects[3]
                
                # Compute new debris
                new_debris = generate_debris_with_margin(object1, object2, margin)
                total_debris_generated_this_epoch += len(new_debris)

                # Add new debris to the total objects array
                objects_fast = np.concatenate((objects_fast, new_debris), axis=0)

                # update the rotation matrices for the new debris
                for _ in new_debris:
                    new_matrix = matrices[random.randint(0, len(matrices) - 1)]  
                    matrices = np.concatenate((matrices, [new_matrix]), axis=0)

                # Save the collision data
                collisions.append([object1, object2, time])
                added_debris.append([new_debris, time])

            #print(f"collision detected, in epoch {time}, number of collisions: {len(collision_pairs)}, number of debris generated: {total_debris_generated_this_epoch}\n")
            
        if (
            post_collision and
            frequency_new_debris != None
            and (time - epoch) % (frequency_new_debris * timestep) == 0
        ):
            post_collision = False
            print(f"post_collision object_fast shape: {objects_fast.shape}, matrices shape: {matrices.shape}")
            objects_fast, matrices, falldown_number = debris_falldown(objects_fast, matrices, INI_NUMBERS + INI_DEBRIS)
            if falldown_number != 0:
                print(f"Debris_falldown detected, in epoch {time}, number of debrits away: {falldown_number} \n")
            else:
                print(f"No Debris_falldown, in epoch {time} \n")

            # objects_fast, matrices = launch_satellites(
            #     objects_fast, matrices, time
            # )

            # objects_fast, matrices, new_debris = random_debris(
            #     objects_fast, matrices, time, percentage
            # )
            # added_debris.append([new_debris, time])

            if draw:
                view.make_new_drawables(objects_fast)
        
        if len(collision_pairs) != 0 or falldown_number != 0:
            file_sys.write(time, current_time, len(collision_pairs), total_debris_generated_this_epoch, falldown_number, file_sys.SIMU_RESULT_PATH, f"Group_{group}")
        
        if draw:
            if (time - epoch) % (frequency_new_debris * timestep) == 0:
                view.make_new_drawables(objects_fast)
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
        margin=5000,
        endtime=100_000,
        timestep=5,
        epoch=1675209600.0,
        probability=0,
        percentage=0,
        frequency_new_debris=40,
    )

    # Save the data to "sim_data" in the correct group folder.

    with open(f"sim_data/group_{objects[0][12]}/parameters.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(
            ["group", "epoch", "endtime", "timestep", "probabilty", "precentage"]
        )
        write.writerows(parameters)

    with open(f"sim_data/group_{objects[0][12]}/collisions.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["object1", "object2", "time"])
        write.writerows(collisions)

    with open(f"sim_data/group_{objects[0][12]}/debris.csv", "w") as csvfile:
        write = csv.writer(csvfile)
        write.writerow(["number_debris", "time"])
        write.writerows(added_debris)

    print(f"\ngroup {group} done running")