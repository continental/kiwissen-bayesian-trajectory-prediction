# Bayesian Continual Learning for Prior Knowledge Integration

Framework for integrating prior knowledge into trajectory prediction models via Bayesian Continual Learning.

## Getting started

### Installing

1. Clone this repository
2. Download [NuScenes Trajectory Prediction Dataset](https://www.nuscenes.org/nuscenes#download)
3. Download [Trajectory Data](https://www.nuscenes.org/public/nuscenes-prediction-challenge-trajectory-sets.zip) and put into data directory
4. Install Dependencies
    1. pip install requirements.txt
    2. pip install requirements_no_deps.txt --no-deps
    3. For GPU Use: cudatoolkit=11.2.2, cudatoolkit-dev=11.2.2, cudnn=8.2.1.32
  
### Running

1. Run with bayes_covernet/main.py as main entrypoint
2. Edit config.gin to match your needs/setup.
3. First start should use -c and -p for populating the cache. See --help for more arguments and description.

For additional arguments, see `python main.py --help`. Running in HEBO optimization mode (-o3) requires an additional `pip install HEBO==0.3.2`

### Experiments

Configuration files for reruning experiments can be found under `bayes_covernet\config`. Please see the paper for hyperparameter configurations. GVCL and Transfer runs require setting a knowledge integrated model checkpoint. GVCL uses a VI_Prior model, set as `Trainer.prior_model`. Transfer uses a Det_Pretrain model, set as `Trainer.load_model`. 

## Model

The provided software computes a trajectory prediction model under consideration of a prior distribution. Currently available are:

### Prior Knowledge

- Discrete Driveable: Discrete set of trajectories, as defined by [CoverNet](https://arxiv.org/abs/1911.10298), conditioned on all keypoints beeing driveable, as defined by the dataset. 

### Prediction Model

- Deterministic [CoverNet](https://arxiv.org/abs/1911.10298)
- Mean-Field Variational [CoverNet](https://arxiv.org/abs/1911.10298)

### Knowledge Integration

- [Generalized Variational Continual Learning](https://openreview.net/pdf?id=_IM-AfFhna9)
- [Transfer Learning](https://arxiv.org/abs/2006.04767)
- [Multitask Learning](https://arxiv.org/abs/2006.04767)
