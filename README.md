# OrbitSmash
Developers: Jenna de Vries, Kato Schmidt, Derk Niessink

## User manual

We developed a simulation for modeling satellites and space debris. In this
simulation we use satellite and debris data in low-Earth orbit (LEO). The data
consist of more than 11000 objects, that is why for performance reasons, we
divided the data in 100 groups. The groups are determined by the semi-major axes of
the objects, because objects that do not have a similar semi-major axis will
never collide.

Our goal was to simulate future space collisions. For this simulation we needed 
to run the simulated time of 10 years and for all groups. This took multiple hours
and we ran the simulation in parallel on the UvA computercluster. With the data
generated in these runs we plotted the graph in our scientific poster. It takes
too long to reproduce this graph, that is why we provided an animation that can
be reproduced when following the steps below.

### Group 19 simulation
![](https://github.com/JennaVries/Project-Minor-Computational-Science/blob/submit_version/earth.gif)


### Running the simulation

1. Clone the repository
2. Create a virtual environment using [uv](https://docs.astral.sh/uv/):
```
$ uv venv
```
3. Activate the virtual environment
4. Install required packages and run the sim with a desired group number: 
```
$ uv pip install -r requirements.txt
$ cd sim
$ python main.py [Group number] [view]
```

An animation can be shown in the browser when adding the second argument "view".

So for example running group 19 with the the animation:
```
$ python main.py 19 view
```
And without the animation:                             
```
$ python main.py 19
```

### In the animation

* Red spheres indicate debris and white satellites.
* Rotate by dragging while holding right click and zoom with mousewheel (easier to control with mouse!).


### After the simulation

* Data will be saved in `sim_data` in the corresponding group number folder.
* A ZeroDivisionError will be printed in the terminal. There is nothing to do
about this and is caused by the fact that vpython (the animation library) has not
updated since the newest version of Python was released.

## For developers

Profiling:
* pip install gprof2dot
* python -m cProfile -o data/profile.stats sim/main.py
* gprof2dot data/profile.stats -f pstats > data/profile.dot
* dot -Tpng data/profile.dot -o data/profile.png
* open data/profile.png
