# Orion - Automated Target Recognition of Military Vehicles

🛰️ A deep-learning–based system for automated detection and classification of military vehicles in video data. Orion integrates visual recognition, motion analysis, and tracking modules to provide real-time situational awareness in complex environments.

<div align="center">
  <img src="docs/imgs/tank_tracking.gif" width="640"/>
</div>

## Documentation

**The documentation for Orion is available [here](https://jonasrenault.github.io/orion/).**

## Installation

Orion requires a recent version of python: ![python_version](https://img.shields.io/badge/Python-%3E=3.12-blue).

### Install from github

Clone the repository and install the project in your python environment, either using `pip`

```bash
git clone https://github.com/jonasrenault/orion.git
cd orion
pip install --editable .
```

or [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/jonasrenault/orion.git
cd orion
uv sync
```

## Usage

### Command-line

When you install Orion in a virtual environment, it creates a CLI script called `orion`. Run

```bash
orion --help
```

to see the various commands available (or take a look at the [documentation](https://jonasrenault.github.io/orion/) for examples).

## Contents

- The [orion](./orion/) directory contains the source code used to fetch and format datasets for fine-tuning a YOLO12 model for object detection.
- The [resources](./resources/) directory contains video samples for vehicle detection task.
- The [notebooks](./notebooks/) directory contains exemple notebooks on how to
  1. [Prepare](./notebooks/01_Prepare.ipynb) a custom dataset of images annotated for automatic target recognition of military vehicles.
  2. [Train](./notebooks/02_Train.ipynb) fine-tune a YOLO12 model using the prepared dataset.
  3. [Evaluate](./notebooks/03_Evaluate.ipynb) a fine-tuned model on a realistic test dataset.

## Run the notebooks

To run the notebooks, start a jupyter lab server with

```bash
jupyter lab
```

and open one of the notebooks in the `notebooks` directory.
