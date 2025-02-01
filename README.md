# Space Debris Dynamics Simulation

## Overview
This repository hosts a dynamic simulation of space debris to study the onset of Kessler Syndrome. The simulation incorporates a probabilistic collision model and a network analysis approach to explore critical transitions within a modeled debris field in Earth orbit. The project uses various computational methods to simulate orbital mechanics, collision probabilities, and debris evolution over time, employing statistical analysis and visualizations to compare the outcomes under various scenarios.

[Kessler Syndrome Simulation](Kessler.gif)


## Authors
- ....
- ....
- ....
- ....

## Dependencies
This project requires the following Python libraries:
- **numpy**: For numerical operations.
- **matplotlib**: For creating visualizations.
- **pandas**: For data manipulation and analysis.
- **networkx**: For creating and manipulating complex network structures.
- **numba**: For accelerating numerical algorithms.
- **vpython**: For real-time 3D animations.
- **scipy**: For scientific and technical computing.

Additionally, standard Python libraries such as `sys`, `os`, `random`, `time`, and `itertools` are used for various helper functions.

## Installation

### Clone the Repository
```bash
git clone https://github.com/zoeazra/ComplexSystemsProject.git
cd space-debris-simulation
```

### Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Unix or MacOS
venv\Scripts\activate  # On Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

Navigate to the simulation directory and execute the main simulation script. Specify a group number to target specific data subsets. Optionally, add `view` to enable visual animations displayed in a web browser.

```bash
cd sim
python main.py [Group number] [view]
```

### Example Commands
- To run the simulation for group 0 without visualization:
  ```bash
  python main.py 0
  ```

- To run the simulation with visualization for group 0:
  ```bash
  python main.py 0 view
  ```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.


