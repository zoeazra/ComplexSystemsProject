"""
"main.py"

Run this module to start the simulation.

Usage:
-> python main.py [orbit number (between 0 and 99)] [view (optional)]

* If "view" is given, an animation will start up in the browser.
* Data will be saved to "sim_data" in the folder corresponding to the group number.
"""


import sys
import csv
from model import *
from model_network import run_network_sim
from data_cleaning import data_array, all_groups

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

    # parameters, collisions, added_debris = run_sim(
    #     objects,
    #     group,
    #     draw,
    #     margin=5000,
    #     endtime=100_000,
    #     timestep=5,
    #     epoch=1675209600.0,
    #     probability=0,
    #     percentage=0,
    #     frequency_new_debris=40,
    # )

    parameters, collisions, added_debris = run_network_sim(
        objects,
        group,
        draw,
        margin=5000,
        endtime=100_000,
        timestep=5,
        epoch=1675209600.0,
        collision_probability=0.001,
        mass_debris_probability= 0.01,
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