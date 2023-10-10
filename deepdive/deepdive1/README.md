# GBSV (Grundlagen der Bild- und Signalverarbeitung)

J. Hartmann, S. Suter

This repository provides material for the FHNW gbsv module.

Deep Dive 1:
* [Deep Dive 1 Notebook 1](Deep Dive 1/gbsv_deep-dive1_part1.ipynb)
* [Deep Dive 1 Notebook 1](Deep Dive 1/gbsv_deep-dive1_part2.ipynb)
* [Requirements](Deep%20Dive%201/requirements.txt)
* [Images](Deep%20Dive%201/Images/)
* [Sounds](Deep%20Dive%201/Sounds/)

## Deep Dive 1: Information and Setup

Because sound is an important type of signal we recommend to bring headphones to the first deep dive.

To be ready to follow the deep dive set up the environment in advance.
Create a new environment with conda and the given requirements by executing the following:

```bash
conda create -n <gbsv_env_name> python=3.11
conda activate <gbsv_env_name>
pip install -r requirements.txt
```

### If you use Linux
If you use Linux please ensure that you installed the package ```libportaudio2``` to your system:
```bash
sudo apt-get install libportaudio2
```