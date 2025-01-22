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
