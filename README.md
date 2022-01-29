# Inverse Constraint RL for Auto-Driving

## commonroad_rl
```
mkdir ./environment
```
This project applies the commonroad_rl environment, please clone their project into to folder ./environment. please refer to the environment setting at
''https://gitlab.lrz.de/tum-cps/commonroad-rl/-/tree/master/''

## stable_baseline3
This project utilizes some implementation in stable_baseline3, but no worry, I have included their code into this project


## Running
To run the code, 

```
# create your model saving dir
mkdir ./save_model
mkdir ./save_model/PPO-highD/
mkdir ./evaluate_model

# start the training
source your-conda-activate
conda activate your-env
cd ./interface
python train_commonroad_ppo.py ../config/train_ppo_highD.yaml -d 1  # for debug mode
python train_commonroad_ppo.py ../config/train_ppo_highD.yaml # for running
```
