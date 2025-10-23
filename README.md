# Orion - Automated Target Recognition of Military Vehicles

🛰️ A deep-learning–based system for automated detection and classification of military vehicles in video data. Orion integrates visual recognition, motion analysis, and tracking modules to provide real-time situational awareness in complex environments.

## Documentation

**The documentation for Orion is available [here](https://jonasrenault.github.io/adomvi/).**

## Install

Orion requires a recent version of python: ![python_version](https://img.shields.io/badge/Python-%3E=3.12-blue).

### Install from github

Clone the repository and install the project in your python environment, either using `pip`

```bash
git clone https://github.com/jonasrenault/adomvi.git
cd orion
pip install --editable .
```

or [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/jonasrenault/adomvi.git
cd orion
uv sync
```

## Usage

### CLI

When you install Orion in a virtual environment, it creates a CLI script called `orion`. Run

```bash
orion --help
```

to see the various commands available (or take a look at the [documentation](https://jonasrenault.github.io/adomvi/) for examples).

## Contents

- The [orion](./adomvi/) directory contains the source code used to fetch and format datasets for training a Yolov8 model for object detection.
- The [resources](./resources/) directory contains video samples for vehicle detection task.
- The [notebooks](./notebooks/) directory contains exemple notebooks on how to
  1. [Prepare](./notebooks/01_Prepare.ipynb) a custom dataset of images annotated for automatic detection of military vehicles.
  2. [Train](./notebooks/02_Train.ipynb) train a Yolov8 model using the prepared dataset.
  3. Run [tracking](./notebooks/03_Track.ipynb) using the trained model on a sample video.
  4. Fine tune [Dreambooth](./notebooks/04_DreamboothFineTuning.ipynb) to generate images of a tank.

## Run the notebooks

To run the notebooks, start a jupyter lab server with

```bash
jupyter lab
```

and open one of the notebooks in the `notebooks` directory.

## TODO

- [ ] add train, track and evaluate notebooks with associate commands
- [ ] add dreambooth logic
- [ ] add google image scraper code with command
- [ ] add documentation and examples with mkdocs
- [ ] update repo name
- [ ] update url for google dataset
- [ ] upload roboflow dataset to github and update roboflow url
