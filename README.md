# Space Debris Simulation

This repository contains a simulation model for studying the dynamics of space debris in Earth orbit, with a focus on the development of the Kessler Syndrome. The model incorporates both probabilistic collision dynamics and network-based analyses to explore critical phase transitions in the debris field.

## Prerequisites

- Python 3.8 or higher
- Virtual environment

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/zoeazra/ComplexSystemsProject.git
   cd space-debris-simulation
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   uv venv
   source venv/bin/activate  # On Unix or MacOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

Navigate to the simulation directory, then execute the main simulation script by specifying a group number. Optionally, add `view` to enable visual animations in the browser.

```bash
cd sim
python main.py [Group number] [view]  # Replace [Group number] with a specific number
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

## Outputs

- **Console Logs:** Displays progress, including initiation and completion of simulation steps, and logs any collision events.
- **Animation:** (Optional) Visual representation of orbital paths and collisions, accessible via a web browser if the `view` argument is used.

## Contributing

Contributions are welcome. Please fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact

- **Your Name** - [YourGitHub](https://github.com/YourGitHub)