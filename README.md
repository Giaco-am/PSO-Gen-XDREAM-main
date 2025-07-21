# PSO-Gen-XDREAM

A novel optimization algorithm for the XDREAM framework for neural activity maximization in deep neural networks, combining Particle Swarm Optimization (PSO) with genetic algorithms. This hybrid algorithm is then tested against a genetic algorithm and CMA-ES.

## Overview

PSO-Gen-XDREAM is an algorithm that extends the [XDREAM](https://klab.tch.harvard.edu/publications/PDFs/gk7791.pdf) (EXtending DeepDream with real-time evolution for activation maximization) methodology. This framework allows researchers to systematically explore and maximize neural activity patterns by synthesizing images that maximally activate a given target unit of a neural network.

## Installation

### Requirements

- Python >= 3.10
- CUDA-compatible GPU (strongly recommended)

### Setup

1. **Clone the repository:**
```bash
git clone <PSO-Gen-XDREAM-main-url>
cd PSO-Gen-XDREAM-main
```

2. **Install the package:**
```bash
pip install -e .
```

This will automatically install all required dependencies.

### Local Configuration

Populate the local settings in `src/xdream/experiment/local_settings.json` to define the paths:

```json
{
    "out_dir": "/path/to/output/directory",
    "weights": "/path/to/generator/weights", 
    "dataset": "/path/to/imagenet/dataset",
   
}
```
where `out_dir` is the path to the output directory, `weights` is the path to the DeepSim weights (downloadable from [here](https://drive.google.com/drive/folders/1sV54kv5VXvtx4om1c9kBPbdlNuurkGFi) choosing the `fc7.pt` variant), and `dataset` is the path to the mini-imagenet dataset (downloadable from [here](https://www.kaggle.com/datasets/arjunashok33/miniimagenet)), please note that the `inet_labels.txt` file is to be placed in the same directory as the dataset.




### Using the Demo Notebook

The included `demo.ipynb` provides a full implementation example of a Maximize Activity Experiment using the PSO-Gen-XDREAM framework. To use it:



1. **Configure the local setting path** in the first cell to match your local setup

2. **Run the experiment** 

3. **Visualize results** using the provided plotting functions

## Framework Architecture

### Algorithm Components

#### 1. **Generator** (`src/xdream/core/generator.py`)
- **DeePSiMGenerator**: Converts latent codes (the subjects of the optimization process) to images using pre-trained generators


#### 2. **Optimizer** (`src/xdream/core/optimizer.py`)
- **CMAESOptimizer**: Covariance Matrix Adaptation Evolution Strategy
- **GeneticOptimizer**: Traditional genetic algorithm implementation  
- **HybridOptimizer**: Novel PSO-GA hybrid approach + K-Means clustering

#### 3. **Subject** (`src/xdream/core/subject.py`)
- **TorchNetworkSubject**: Wraps PyTorch models for neural activity recording


#### 4. **Scorer** (`src/xdream/core/scorer.py`)
- **ActivityScorer**: Evaluates neural activation levels
- It has configurable unit and layer reduction strategies

#### 5. **Experiment** (`src/xdream/core/experiment.py`)
- **MaximizeActivityExperiment**: Main experiment class for activity maximization


### Hybrid PSO Optimizer Features

The novel Hybrid PSO optimizer combines Particle Swarm Optimization with Genetic Algorithms:

#### **Hybrid Architecture**
- **GA Phase**: Provides global exploration and generates diverse population across search space
- **Clustering Phase**: Groups similar solutions in search space
- **Selective PSO Phase**: Fine-tunes only the most promising solutions in each region
- **Velocity Reset**: After GA operations, velocities are reset based on personal best directions

#### **Dynamic Scheduling System**
- **Multi-Stage PSO phase occurency**: 
  - Early phase (< 60% iterations): Primarily GA with PSO every 5 steps
  - Middle-Late phase (60-85% iterations): Balanced approach with PSO every 2 steps  

- **Progress-Adaptive Intervals**: Configurable thresholds via `first_PSO_interval=0.6` and `second_PSO_interval=0.85`

#### **Intelligent Parameter Adaptation**
- **Stagnation Detection**: Monitors fitness improvement using `stagnation_threshold=0.01`
- **Diversity Monitoring**: Tracks population diversity with `diversity_threshold=0.01`
- **Dynamic Hyperparameter Adjustment**:
  - **Cognitive Component**: Adaptive scaling with `less_cog_factor=0.995` and `more_cog_factor=1.05`
  - **Mutation Parameters**: Dynamic mutation rate (`more_mut_rate=1.005`, `less_mut_rate=0.995`) and size (`less_mut_size=0.955`) adjustment
  - **Performance-Based Triggers**: Parameters adapt based on recent fitness history analysis

#### **Advanced PSO Configuration**
- **Linearly Adaptive Inertia**: Dynamic inertia weight adjustment between `inertia_max` and `inertia_min`
- **Social Network Topology**: Configurable informant networks with `num_informants=20`
- **Velocity Constraints**: Bounded velocity updates with `v_clip=0.15`
- **Enhanced Exploration-Exploitation Balance**: Cognitive (`cognitive=2.5`) and social (`social=2.0`) component weighting

#### **Genetic Algorithm Integration**
- **Multi-Parent Crossover**: Supports `n_parents=4` with clone allowance (`allow_clones=True`)
- **Elite Preservation**: Top-k selection strategy with `topk` individuals preserved
- **Temperature-Controlled Selection**: Softmax-based parent selection with `temp=1.2` and decay (`temp_factor=0.98`)
- **Adaptive Mutation Strategy**: Dynamic mutation size and rate with intelligent scaling


This hybrid approach leverages the complementary strengths of both algorithms: GA's robust exploration capabilities for escaping local optima, and PSO's efficient exploitation for fine-tuning to promising regions of the search space.

### Multi-Experiment Analysis

Run systematic parameter searches:

1. In `src/experiments/MaximizeActivity/run/multirun_arguments.py` set the multi-experiment name in NAME, give the other hyper-params a look and adjust them as needed.
2. On your terminal run:
```bash
python multirun_arguments.py
```
from the directory this file is located in. This will generate a list of inputs you can choose from:
1. If you select '1', the multiexperiment which compares the hybrid optimizer against the genetic and the CMA-ES optimizers will be run.
2. If you select '2', the multiexperiment which performs a parameter search for the hybrid optimizer will be run.

Once you select a choice, go to the generated `run/cmd2exec.txt` copy it and paste it in your terminal to run the experiment. The results will be saved as a `data.pkl` in the `output/` directory you specified in the `local_settings.json` file. 

Afterwards, you can visualize the results using the provided plotting functions in `demo.ipynb`.


### Acknowledgements
This project is an XDREAM re-implementation made by Lorenzo Tausani [@LorenzoTausani](https://github.com/LorenzoTausani), Sebastiano Quintavalle [@Quinta13](https://github.com/Quinta13), Paolo Muratore [@myscience](https://github.com/myscience) and Giacomo Amerio [@Giaco-am](https://github.com/Giaco-am).


