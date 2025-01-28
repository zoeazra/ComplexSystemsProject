import numpy as np
from numba import jit
import random
import file_sys
from view import View
import time as tf
from tqdm import tqdm
import numpy as np
import sys
from scipy.spatial.transform import Rotation
import model as md_classic
import itertools
from numba import njit, prange



# standard gravitational parameter = G * M
mu = 6.6743 * 10**-11 * 5.972 * 10**24  # m**3 * s**-2

# define a variable of the length of the initial objects
INI_NUMBERS = 49

def run_network_sim(
    objects: np.ndarray,
    group: int,
    draw: bool,
    margin: float,
    endtime: float,
    timestep: float,
    epoch: float,
    collision_probability: float,
    mass_debris_probability: float,
    frequency_new_debris: int = None,
) -> tuple[list, list, list]:

    objects = objects[0 : INI_NUMBERS-1] # pick up the first INI_NUMBERS objects from the real data
    md_classic.initialize_positions(objects, epoch)
    objects_fast = md_classic.fast_arr(objects)
    matrices = np.array([object[11] for object in objects])

    if draw:
        view = View(objects_fast)

    parameters, collisions, added_debris = [], [], []
    current_time = tf.strftime("%Y%m%d-%H%M%S")

    for time in tqdm(
        range(int(epoch), int(epoch + endtime), timestep),
        ncols=100,
        desc=f"group: {group}",  # tqdm for the progress bar.
    ):  
    #for time in range(int(epoch), int(epoch + endtime), timestep):
        #print(f"\nGroup {group} process running at time {time}.")
        md_classic.calc_all_positions(objects_fast, matrices, time)

        if len(objects_fast) > 5000:
            print(f"\nGroup {group} process killed.")
            sys.exit()

        # Use network model to check for collisions, with the Probability of collision -- P
        collision_pairs = pick_up_collision_pairs(objects_fast, collision_probability)
        # print(f"collision_pairs create: {len(collision_pairs)}")

        total_debris_generated_this_epoch = 0
        falldown_number = 0

        if len(collision_pairs) != 0:
            
            for collided_objects in collision_pairs:
                object1, object2 = collided_objects[0], collided_objects[1]
                
                # Compute new debris
                new_debris = generate_debris_with_probability(object1, object2, mass_debris_probability, margin)
                if len(new_debris) == 0 or new_debris[0][0] == 0:
                    continue
                
                print(f"A massive debris detected, in epoch {time} \n")
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
            frequency_new_debris != None
            and (time - epoch) % (frequency_new_debris * timestep) == 0
        ):  # Add new debris or satellites at timesteps indicated by frequency_new_debris.
            
            # objects_fast, matrices, falldown_number = debris_falldown(objects_fast, matrices)
            # if falldown_number != 0:
            #     print(f"Debris_falldown detected, in epoch {time}, number of debrits away: {falldown_number} \n")

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
        [objects[0][12], epoch, endtime, timestep, collision_probability, mass_debris_probability]
    )

    return parameters, collisions, added_debris

def pick_up_collision_pairs(objects_fast, probability):
    """
    Pick up the collision pairs from the network model, which means there is one edge created between two nodes.
    """
    # Pick up the collision pairs from the network model
    N = len(objects_fast)
    num_possible_edges = N * (N - 1) // 2  # pick up the number of possible edges
    num_collisions = int(probability * num_possible_edges)  # pick up the number of collisions

    # randomly pick up the collision pairs
    all_edges = list(itertools.combinations(objects_fast, 2))
    sampled_edges = np.random.choice(len(all_edges), size=num_collisions, replace=False)
    choosed_pairs = [all_edges[i] for i in sampled_edges]

    return choosed_pairs

@jit(nopython=True, parallel=True)
def generate_debris_with_probability(object1: np.ndarray, object2: np.ndarray, probability: float, margin: float) -> np.ndarray:
    """
    Generate the debris in a probabilistic way, which means the collision has a low probability to generate those could be regarded as the node into the network model.
    Those debris lower than a thredshold mass will be ingored from this network model.

    Parameters:
        object1: np.ndarray
            Array representing the first object in collision:
            ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']
        object2: np.ndarray
            Array representing the second object in collision:
            ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']
        probability: float
            Probability of generating node debris, should be low.
        margin: float
            Margin for positional adjustments.

    Returns:
        np.ndarray
            Array of new debris generated:
            [['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']]
    """
    # Random number for debris generation
    num_debris = 0
    if np.random.rand() < probability:
        num_debris = 1
    else:
        return np.zeros((0, 7), dtype=np.float64)

    # Preallocate array for debris with explicit types: [int, float, float, int, float, float, float]
    new_debris = np.zeros((num_debris, 7), dtype=np.float64)

    # Randomize new parameters for debris
    g = np.random.rand()  # Random number for semimajor-axis adjustment
    new_semi_major_axis = object1[2] + ((g * 200) - 100)
    
    # Adjust mean anomaly
    new_mean_anomaly = object1[1] + 180
    if new_mean_anomaly > 360:
        new_mean_anomaly -= 360

    # Parallel loop to initialize debris
    for i in prange(num_debris):
        new_debris[i, 0] = object1[0]  # EPOCH (int)
        new_debris[i, 1] = new_mean_anomaly  # MEAN_ANOMALY (float)
        new_debris[i, 2] = new_semi_major_axis  # SEMIMAJOR_AXIS (float)
        new_debris[i, 3] = 1  # SATELLITE/DEBRIS_BOOL (int)
        new_debris[i, 4] = object1[4] + np.random.uniform(margin, 2 * margin)  # pos_x (float)
        new_debris[i, 5] = object1[5] + np.random.uniform(margin, 2 * margin)  # pos_y (float)
        new_debris[i, 6] = object1[6] + np.random.uniform(margin, 2 * margin)  # pos_z (float)

    return new_debris
