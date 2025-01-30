"""
"model.py"

This module contains all model functions. Functions which have to be called
every time step are accelerated using Numba's jit decorator. The decorator
ensures that the these functions will be compiled in C code.
"""

import numpy as np
from numba import jit
import random
import file_system.file_sys as file_sys
from view import View
import time as tf
from tqdm import tqdm
import numpy as np
import sys
from scipy.spatial.transform import Rotation
from numba import njit, prange

# standard gravitational parameter = G * M
mu = 6.6743 * 10**-11 * 5.972 * 10**24  # m**3 * s**-2

# define a variable of the length of the initial objects
INI_NUMBERS = 49

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

    objects = objects[0 : INI_NUMBERS-1] # pick up the first INI_NUMBERS objects from the real data
    initialize_positions(objects, epoch)
    objects_fast = fast_arr(objects)
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
        calc_all_positions(objects_fast, matrices, time)

        if len(objects_fast) > 500000000:
            print(f"\nGroup {group} process killed.")
            sys.exit()

        # issue 1, make a list of all the objects that are colliding, not just the first collision
        # issue 2, should update the matrices for the new debris
        collision_pairs = check_collisions_optimized(objects_fast, margin)
        total_debris_generated_this_epoch = 0
        falldown_number = 0

        if len(collision_pairs) != 0:
            
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
            frequency_new_debris != None
            and (time - epoch) % (frequency_new_debris * timestep) == 0
        ):  # Add new debris or satellites at timesteps indicated by frequency_new_debris.
            
            objects_fast, matrices, falldown_number = debris_falldown(objects_fast, matrices)
            if falldown_number != 0:
                print(f"Debris_falldown detected, in epoch {time}, number of debrits away: {falldown_number} \n")

            # objects_fast, matrices = launch_satellites(
            #     objects_fast, matrices, time
            # )

            objects_fast, matrices, new_debris = random_debris(
                objects_fast, matrices, time, percentage
            )
            added_debris.append([new_debris, time])

            if draw:
                view.make_new_drawables(objects_fast)
        
        # if len(collision_pairs) != 0 or falldown_number != 0:
        #     file_sys.write(time, current_time, len(collision_pairs), total_debris_generated_this_epoch, falldown_number, file_sys.SIMU_RESULT_PATH, f"Group_{group}")
        
        if draw:
            if (time - epoch) % (frequency_new_debris * timestep) == 0:
                view.make_new_drawables(objects_fast)
            view.draw(objects_fast, time - epoch)

    parameters.append(
        [objects[0][12], epoch, endtime, timestep, probability, percentage]
    )

    return parameters, collisions, added_debris

def initialize_positions(objects: np.ndarray, epoch=1675209600.0):
    """
    Initialize all objects in the given array to the same given epoch by
    adjusting object's true anomaly.

    objects: array of objects to be calibrated in the following form:
        -> ['EPOCH', 'INCLINATION', 'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER',
       'MEAN_ANOMALY', 'NORAD_CAT_ID', 'SEMIMAJOR_AXIS', 'OBJECT_TYPE',
       'RCS_SIZE', 'LAUNCH_DATE', 'positions', 'rotation_matrix', 'groups'].
    epoch: desired Julian date in seconds (default = Monday 1 November 2021 13:00:01).
    """
    for object in objects:
        initialized_anomaly = calc_new_anomaly(epoch, object[0], object[4], object[6])
        object[4] = initialized_anomaly
        object[0] = epoch


def random_debris(
    objects: np.ndarray,
    matrices: np.ndarray,
    time: float,
    percentage: float,
) -> tuple[np.ndarray, np.ndarray, int]:

    """
    Add a certain amount, given by the percentage of the existing debris, of
    debris with random orbits and positions. The new debris is added to the
    objects and its random rotation matrix to matrices.

    objects: array of all objects of the form:
    -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL',
    'pos_x', pos_y', 'pos_z'].
    matrices: array of all rotation matrices of the objects.
    time: current simulation times.
    percentage: desired percentage of the number of existing objects to add.

    Returns a tuple of the new objects array, matrices array and the number
    of new debris.
    """

    # n_new_debris = np.ceil(len(objects) * (percentage / 100))

    for _ in range(int(1)):
        mean_anomaly, semimajor_axis, matrix = random_params(objects)
        matrices = np.append(matrices, matrix, axis=0)
        pos = new_position(time, time + 1, mean_anomaly, semimajor_axis, matrices[-1])
        new_debris = np.array(
            [[time, mean_anomaly, semimajor_axis, 1, pos[0], pos[1], pos[2]]]
        )

        objects = np.append(objects, new_debris, axis=0)
    return objects, matrices, int(1)

def launch_satellites(
    objects: np.ndarray,
    matrices: np.ndarray,
    time: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    mean_anomaly, semimajor_axis, matrix = random_params(objects)
    matrices = np.append(matrices, matrix, axis=0)
    pos = new_position(time, time + 1, mean_anomaly, semimajor_axis, matrices[-1])
    new_satellite = np.array(
        [[time, mean_anomaly, semimajor_axis, 0, pos[0], pos[1], pos[2]]]
    )

    objects = np.append(objects, new_satellite, axis=0)
    return objects, matrices

def random_params(objects) -> tuple[float, float, np.ndarray]:
    """
    Returns random object parameters.

    objects: array of all objects of the form:
    -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL',
    'pos_x', pos_y', 'pos_z'].

    Return a tuple of the random mean_anomaly in degrees, random semimajor-axis
    in meters (chosen from existing objects) and rotation matrix.
    """
    R = Rotation.from_euler(
        "zxz",
        [
            -np.random.uniform(0, 360),
            -np.random.normal(0, 360),
            -np.random.uniform(0, 360),
        ],
        degrees=True,
    )
    mean_anomaly = np.random.uniform(0, 360)
    semimajor_axis = objects[np.random.randint(len(objects))][2]
    return mean_anomaly, semimajor_axis, np.array([R.as_matrix()])


@jit(nopython=True)
def calc_new_anomaly(
    time: float, epoch: float, mean_anomaly: float, semimajor_axis: float
) -> float:
    """
    Calculate the new anomaly of an object at a specific Julian date in
    seconds.

    time: Julian date in seconds of the desired anomaly.
    epoch: Julian date in seconds.
    mean_anomaly: anomaly corresponding to the object's epoch in degrees.
    semimajor_axis: semimajor-axis of the object's orbit in meters.
    """
    time_delta = time - epoch
    return mean_anomaly + time_delta * np.sqrt(mu / semimajor_axis**3)


@jit(nopython=True)
def new_position(
    time: float,
    epoch: float,
    mean_anomaly: float,
    semimajor_axis: float,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate the position of an object at specific point in time

    time: time in seconds after object's epoch at which the position will
    computed.
    epoch: time corresponding to the mean anomaly of the object.
    mean_anomaly: anomaly in rad corresponding to the time.
    semimajor_axis: semimajor axis of the orbit.
    rotation_matrix: rotation matrix computed from the 3 orbital angles.

    Returns the 3D position vector (in the Earth frame) of the object at
    the given time.
    """
    true_anomaly = calc_new_anomaly(time, epoch, mean_anomaly, semimajor_axis)
    pos_orbit_frame = (
        np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0]) * semimajor_axis
    )
    return rotation_matrix.dot(pos_orbit_frame)


@jit(nopython=True)
def calc_all_positions(
    objects: np.ndarray, matrices: np.ndarray, time: float
) -> np.ndarray:
    """
    Calculate the new positions of all objects and update the objects array.

    objects: array of objects to be evaluated. An object has to be in the
    following form:
     -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL',
    'pos_x', pos_y', 'pos_z'].
    marices: array of rotation matrices of the objects computed from the 3
    orbital angles.
    time: time at which the positions will be calculated.
    """
    for i in range(len(objects)):

        pos = new_position(
            time,
            epoch=objects[i][0],
            mean_anomaly=objects[i][1],
            semimajor_axis=objects[i][2],
            rotation_matrix=matrices[i],
        )
        objects[i][4], objects[i][5], objects[i][6] = (
            pos[0],
            pos[1],
            pos[2],
        )


@jit(nopython=True)
def check_collisions(
    objects: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Checks for collisions by iterating over all possible combinations,
    by checking if the objects in the combination share a similar position.
    A similar positions means that the distance between the position vectors
    is smaller than the given margin.

    objects: array of objects to be evaluated. An object has to be in the
    following form:
     -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL',
    'pos_x', pos_y', 'pos_z'].
    margin: say that there could be a collision when the distance is smaller
    than this value.

    returns a tuple of the two candidate colliding objects if the distance
    between the vectors is smaller than the margin.
    """
    for i in range(len(objects) - 1):
        for j in range(i + 1, len(objects) - 1):
            if (
                objects[i][3] != 0 or objects[j][3] != 0
            ):  # Satellites cannot collide with other satellites

                pos1 = np.array([objects[i][4], objects[i][5], objects[i][6]])
                pos2 = np.array([objects[j][4], objects[j][5], objects[j][6]])

                if np.linalg.norm(pos1 - pos2) < margin:
                    return objects[i], objects[j]

#@jit(nopython=True)
def check_collisions_optimized(objects: np.ndarray, margin: float) -> list:
    """
    1. Optimized version of the check_collisions function using Numba's parallel, which splits the loop iterations in multi cores.
    2. Avoiding the array slicing.
    3. Meanwhile, collect all the collision pairs then return them in a list.

    Parameters:
        objects (np.ndarray): Array of objects in the form:
            ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']
        margin (float): Collision threshold distance.

    Returns:
        list of tuples: Each tuple contains (index_i, index_j, object_i, object_j)
    """
    collision_pairs = []

    for i in range(len(objects) - 1):
        for j in range(i + 1, len(objects)):
            if objects[i][3] != 0 or objects[j][3] != 0:
                # avoid array slicing, manually extract and calculate the distance
                dx = objects[i][4] - objects[j][4]
                dy = objects[i][5] - objects[j][5]
                dz = objects[i][6] - objects[j][6]
                distance = (dx * dx + dy * dy + dz * dz) ** 0.5
                if distance < margin:
                    collision_pairs.append((i, j, objects[i], objects[j]))
                    
                    # avoid cpu burning, return if the number of collision pairs exceeds 100
                    # THIS IS FOR ALEX CPU
                    # if len(collision_pairs) > 5:
                    #     return collision_pairs
    return collision_pairs

#@jit(nopython=True)
def collision(object1: np.ndarray, object2: np.ndarray) -> np.ndarray:
    """
    Generate debris as a result of a collision.

    object1, object2: The two colliding objects, each in the form:
    -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL',
       'pos_x', 'pos_y', 'pos_z'].

    Returns:
        A NumPy array of debris objects.
    """
    r1 = np.linalg.norm(np.array([object1[4], object1[5], object1[6]])) + 1e-6  # distance from Earth for object 1
    r2 = np.linalg.norm(np.array([object2[4], object2[5], object2[6]])) + 1e-6  # distance from Earth for object 2


    v1 = np.sqrt(mu * ( (2 / r1) - 1 / object1[2])) + 1e-6 # velocity of object 1
    v2 = np.sqrt(mu * ( (2 / r2) - 1 / object2[2])) + 1e-6 # velocity of object 2

    rel_velocity = np.abs(v1 - v2)
    if rel_velocity == 0:
        rel_velocity = 1e-6
    num_debris = max(2, min(int(rel_velocity // 500), 7))  # Debris generated per collision. Check again the factor of 500. Clip in order to keep it in a normal range.

    new_debris = np.zeros((num_debris, 7))  # Preallocate array for debris

    for i in range(num_debris):
        g = np.random.rand()
        new_semi_major_axis = object1[2] + ((g * 200) - 100)
        new_mean_anomaly = object1[1] + np.random.uniform(-30, 30)

        # Wrap anomalies within 0-360
        if new_mean_anomaly > 360:
            new_mean_anomaly -= 360
        elif new_mean_anomaly < 0:
            new_mean_anomaly += 360

        new_debris[i, 0] = object1[0]  # EPOCH
        new_debris[i, 1] = new_mean_anomaly  # Mean anomaly
        new_debris[i, 2] = new_semi_major_axis  # Semi-major axis
        new_debris[i, 3] = 1  # Mark as debris
        new_debris[i, 4] = object1[4] + np.random.uniform(-10, 10)  # pos_x
        new_debris[i, 5] = object1[5] + np.random.uniform(-10, 10)  # pos_y
        new_debris[i, 6] = object1[6] + np.random.uniform(-10, 10)  # pos_z

    return new_debris

def number_of_debris_this_pair(object1: np.ndarray, object2: np.ndarray) -> int:
    """
    Calculate the number of debris generated per collision based on the relative velocity.

    object1: The first object involved in the collision.
    object2: The other object involved in the collision.

    Returns:
        The number of debris generated per collision.
    """

    r1 = np.linalg.norm(np.array([object1[4], object1[5], object1[6]])) + 1e-6  # distance from Earth for object 1
    r2 = np.linalg.norm(np.array([object2[4], object2[5], object2[6]])) + 1e-6  # distance from Earth for object 2


    v1 = np.sqrt(mu * ( (2 / r1) - 1 / object1[2])) + 1e-6 # velocity of object 1
    v2 = np.sqrt(mu * ( (2 / r2) - 1 / object2[2])) + 1e-6 # velocity of object 2

    rel_velocity = np.abs(v1 - v2)
    if rel_velocity == 0:
        rel_velocity = 1e-6
    num_debris = max(2, min(int(rel_velocity // 500), 5))  # Debris generated per collision. Check again the factor of 500. Clip in order to keep it in a normal range.
    # print(f"Number of debris generated per collision: {num_debris}")
    return num_debris

@jit(nopython=True, parallel=True)
def generate_debris_with_margin(object1: np.ndarray, object2: np.ndarray, margin: float) -> np.ndarray:
    """
    Generate debris after collision with adjusted parameters.

    Parameters:
        object1: np.ndarray
            Array representing the first object in collision:
            ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']
        object2: np.ndarray
            Array representing the second object in collision:
            ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']
        margin: float
            Margin for positional adjustments.

    Returns:
        np.ndarray
            Array of new debris generated:
            [['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATELLITE/DEBRIS_BOOL', 'pos_x', 'pos_y', 'pos_z']]
    """
    num_debris = 2 # Fixed number of debris generated per collision
    # num_debris = number_of_debris_this_pair(object1, object2)

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

def debris_falldown(objects: np.ndarray, rotation_matrix: np.ndarray, initial_numbers = INI_NUMBERS):
    """
    Simulate the debris falling down to the Earth.
    """

    # randomly pick up debris from object[initial_numbers:] to remove them, b the uniform distribution
    # sync the rotation_matrix with the objects
    if len(objects) - initial_numbers < 100:
        return objects, rotation_matrix, 0

    # numbers_falldown follow the normal distribution from 0 to len(objects) - initial_numbers
    # the expect is the half of the len(objects) - initial_numbers, and the std is the half of the expect
    numbers_falldown = int(np.random.normal((len(objects) - initial_numbers) / 10, (len(objects) - initial_numbers) / 20))
    
    if numbers_falldown <= 0 or numbers_falldown >= len(objects) - initial_numbers:
        return objects, rotation_matrix, 0

    # make sure the numbers_falldown is graeter than 0 and less than len(objects) - initial_numbers
    assert numbers_falldown > 0 and numbers_falldown < len(objects) - initial_numbers

    # just remove the debris from index initial_numbers to initial_numbers + numbers_falldown
    objects = np.delete(objects, np.arange(initial_numbers, initial_numbers + numbers_falldown), axis=0)
    rotation_matrix = np.delete(rotation_matrix, np.arange(initial_numbers, initial_numbers + numbers_falldown), axis=0)
    print(f"Debris falldown: {numbers_falldown} debris have fallen down to the Earth.")
    return objects, rotation_matrix, numbers_falldown

#@jit(nopython=True)
#def collision(object1: np.ndarray, object2: np.ndarray):
#    """
#    Add a new debris at the position of the objects involved with a adjusted
#    anomaly and semimajor-axis.
#
#    object_involved: np.array of the object to be evaluated and has to be in the
#    following form:
#     -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATTELITE/DEBRIS_BOOL',
#    'pos_x', pos_y', 'pos_z'].
#
#    Returns an array of the same form as above with the adjusted values.
#    """
#    new_debris = list()
#    g = np.random.rand()
#    new_semi_major_axis = object1[2] + ((g * 200) - 100)
#
#    new_mean_anomaly = object1[1] + 180
#    if new_mean_anomaly > 360:
#        new_mean_anomaly -= 360

#    new_debris.append(
#        [
#            object1[0],
#            new_mean_anomaly,
#            new_semi_major_axis,
#            1,
#            -object1[4],
#            -object1[5],
#            -object1[6],
#        ]
#    )

#    return new_debris
