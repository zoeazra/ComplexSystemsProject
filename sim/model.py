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

    n_new_debris = np.ceil(len(objects) * (percentage / 100))

    for _ in range(int(n_new_debris)):
        mean_anomaly, semimajor_axis, matrix = random_params(objects)
        matrices = np.append(matrices, matrix, axis=0)
        pos = new_position(time, time + 1, mean_anomaly, semimajor_axis, matrices[-1])
        new_debris = np.array(
            [[time, mean_anomaly, semimajor_axis, 1, pos[0], pos[1], pos[2]]]
        )

        objects = np.append(objects, new_debris, axis=0)
    return objects, matrices, int(n_new_debris)


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


@jit(nopython=True)
def collision(object1: np.ndarray, object2: np.ndarray):
    """
    Add a new debris at the position of the objects involved with a adjusted
    anomaly and semimajor-axis.

    object_involved: np.array of the object to be evaluated and has to be in the
    following form:
     -> ['EPOCH', 'MEAN_ANOMALY', 'SEMIMAJOR_AXIS', 'SATTELITE/DEBRIS_BOOL',
    'pos_x', pos_y', 'pos_z'].

    Returns an array of the same form as above with the adjusted values.
    """
    new_debris = list()
    g = np.random.rand()
    new_semi_major_axis = object1[2] + ((g * 200) - 100)

    new_mean_anomaly = object1[1] + 180
    if new_mean_anomaly > 360:
        new_mean_anomaly -= 360

    new_debris.append(
        [
            object1[0],
            new_mean_anomaly,
            new_semi_major_axis,
            1,
            -object1[4],
            -object1[5],
            -object1[6],
        ]
    )

    return new_debris
