"""
"model.py"

This module contains all model functions. Functions which have to be called
every time step are accelerated using Numba's jit decorator. The decorator
ensures that the these functions will be compiled in C code.
"""

import numpy as np
from numba import jit

from scipy.spatial.transform import Rotation

# standard gravitational parameter = G * M
mu = 6.6743 * 10**-11 * 5.972 * 10**24  # m**3 * s**-2


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

@jit(nopython=True, parallel=True)
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
                    if len(collision_pairs) > 5:
                        return collision_pairs
    return collision_pairs

@jit(nopython=True)
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

@jit(nopython=True)
def generate_debris_with_margin(object1: np.ndarray, object2: np.ndarray, margin: float) -> np.ndarray:
    """
    Add a new debris at the position of the objects involved with a adjusted
    anomaly and semimajor-axis.

    object_involved: np.array of the object to be evaluated and has to be in the
    following form:
        -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATTELITE/DEBRIS_BOOL',
    'pos_x', pos_y', 'pos_z'].

    Returns an array of the same form as above with the adjusted values.
    """
    num_debris = 2 # Fixed number of debris generated per collision
    new_debris = list()
    g = np.random.rand()
    new_semi_major_axis = object1[2] + ((g * 200) - 100)

    new_mean_anomaly = object1[1] + 180
    if new_mean_anomaly > 360:
        new_mean_anomaly -= 360

    for i in range(num_debris):
        new_debris.append(
            [
                object1[0],
                new_mean_anomaly,
                new_semi_major_axis,
                1,
                object1[4] + np.random.uniform(margin, 2*margin),
                object1[5] + np.random.uniform(margin, 2*margin),
                object1[6] + np.random.uniform(margin, 2*margin),
            ]
        )

    return new_debris

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
