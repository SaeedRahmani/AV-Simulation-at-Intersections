# A Bi-level Framework for Modeling 2D Vehicular Movements and Interactions at Urban Junctions

This project is a comprehensive collection of scripts and modules related to motion primitives and A* search algorithms. It provides a robust framework for creating and visualizing motion primitives for different vehicle models, and includes a variety of scenarios and environments for testing and development.

## packages
numpy
matplotlib
cvxpy

brew install ffmpeg

## Project Structure

- `main/`: This directory contains the core scripts and modules of the project.
    - `create_motion_primitives_bicycle_model.py`: This script generates motion primitives for a bicycle model.
    - `create_motion_primitives_prius.py`: This script generates motion primitives for a Prius model.
    - `envs/`: This directory contains scripts for various environments in which the motion primitives can be tested.
    - `lib/`: This directory contains library scripts, including implementations of the A* search algorithm and scripts for searching motion primitives.
    - `scenarios/`: This directory contains scripts for different scenarios, providing a variety of contexts in which to test the motion primitives.
    - `tests/`: This directory contains unit tests for the project, ensuring the reliability and correctness of the code.
- `data/`: This directory contains data related to the motion primitives, providing a resource for further analysis and development.
- `results/`: This directory is where results from tests and simulations are stored.
- `visualise_mp.ipynb`: This Jupyter notebook provides a visual interface for inspecting the motion primitives.

## Installation

This project uses Poetry for dependency management. To install the project dependencies, run:

```sh
poetry install


To be completed...
