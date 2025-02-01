"""
Data cleaning module for satellite and debris dataset.

Satellite and debris data source: http://astria.tacc.utexas.edu/AstriaGraph/
"""

import pandas as pd
import numpy as np
import os
import datetime
from scipy.spatial.transform import Rotation

# Load dataset from CSV file
dataset = pd.read_csv("../data/satellites.csv")

# removing irrelevant columns
dataset = dataset.drop(
    columns=[
        "CCSDS_OMM_VERS", "COMMENT", "CREATION_DATE", "ORIGINATOR", "OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", "REF_FRAME",
        "TIME_SYSTEM", "MEAN_ELEMENT_THEORY", "EPHEMERIS_TYPE", "CLASSIFICATION_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH", "BSTAR",
        "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", "SITE", "DECAY_DATE", "FILE", "GP_ID", "TLE_LINE0", "TLE_LINE1", "TLE_LINE2",
        "ECCENTRICITY", "MEAN_MOTION", "PERIOD", "APOAPSIS", "PERIAPSIS", "COUNTRY_CODE",
    ]
)


def epoch(df_column):
    """Convert datetime column to epoch timestamps."""

    date_list = list(df_column)
    new_date_list = []

    for data in date_list:
        date, time = data.split("T")
        year, month, day = date.split("-")
        hour, minute, second = time.split(":")
        second = second[0:2]

        new_date_list.append(
            datetime.datetime(
                int(year), int(month), int(day), int(hour), int(minute), int(second)
            ).timestamp()
        )
    return new_date_list


# Select only data in Low Earth Orbit (LEO)
dataset = dataset.sort_values("SEMIMAJOR_AXIS")
dataset = dataset[dataset["SEMIMAJOR_AXIS"] < 8371]
dataset["MEAN_ANOMALY"] = dataset["MEAN_ANOMALY"] * np.pi / 180
dataset["EPOCH"] = epoch(dataset["EPOCH"])
dataset["tuples"] = [(0, 0, 0) for i in range(len(dataset.index))]
dataset["SEMIMAJOR_AXIS"] = dataset["SEMIMAJOR_AXIS"].apply(
    lambda x: x * 1000
)  # Convert to meters

# Create rotation matrices for all objects
matrices = []
for index, row in dataset.iterrows():
    R = Rotation.from_euler(
        "zxz",
        [-row["RA_OF_ASC_NODE"], -row["INCLINATION"], -row["ARG_OF_PERICENTER"]],
        degrees=True,
    )
    matrices.append(R.as_matrix())

dataset["rotation_matrix"] = matrices


# Grouping data by semimajor axis range
linspace = np.linspace(
    min(dataset["SEMIMAJOR_AXIS"]), max(dataset["SEMIMAJOR_AXIS"]), num=100
)
bins = np.digitize(np.array(dataset["SEMIMAJOR_AXIS"]), linspace, right=False)
dataset["groups"] = bins

# Selecting a subset of groups
subgroups = dataset.loc[dataset["groups"].isin([i for i in range(17, 42)])]
linspace_sub = np.linspace(
    min(subgroups["SEMIMAJOR_AXIS"]), max(subgroups["SEMIMAJOR_AXIS"]), num=100
)
bins_sub = np.digitize(np.array(subgroups["SEMIMAJOR_AXIS"]), linspace_sub, right=False)
subgroups_ = subgroups.copy()
subgroups_["groups"] = bins_sub

dataset = subgroups_
# dataset = subgroups_.loc[subgroups_["groups"] != 19] ## ignore this comment

# Remove groups with only one object
small_group = dataset.groupby("groups")["groups"].count() != 1
delete = list(small_group.loc[small_group == False].index)

# Remove groups without debris
grouped = dataset.groupby("groups")
debris_groups = grouped.filter(
    lambda x: all(x["OBJECT_TYPE"].isin(["TBA", "ROCKET BODY", "PAYLOAD"]))
)
no_debris = []
for i in set(debris_groups["groups"]):
    no_debris.append(i)
delete.extend(no_debris)

dataset = dataset[~dataset["groups"].isin(delete)]
group_amount = dataset.groupby("groups")["groups"].count()

# Create data folders for all groups
all_groups = []
for i in group_amount.index:
    all_groups.append(i)
    if not os.path.exists(f"sim_data/group_{i}"):
        os.makedirs(f"sim_data/group_{i}", exist_ok=True)

# Keep only payload objects
dataset = dataset.copy()

# only keep payloads in the dataset
dataset = dataset[dataset["OBJECT_TYPE"] == "PAYLOAD"]

# Create binary column for object type
bool_list = [0 if j == "PAYLOAD" else 1 for _, j in dataset["OBJECT_TYPE"].items()]
dataset["object_bool"] = bool_list

# Final dataset preparation
print("DATA CLEAN fully OBJECT Columns:", dataset.columns.tolist())
data_array = dataset.to_numpy()
