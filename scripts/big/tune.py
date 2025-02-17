import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import torch
num_gpus = torch.cuda.device_count()
LSF_COMMAND = "bsub -q gpuqueue -gpu " +\
"\"num=1:j_exclusive=yes\" -R \"rusage[mem=40] span[ptile=1]\" -W 0:10 -Is "

PYTHON_COMMAND =\
"python3 scripts/big/run.py"

if num_gpus > 0:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=6, num_gpus=num_gpus)
    resource = {"cpu": num_gpus, "gpu": num_gpus}
else:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus)
    resource = {"cpu": num_cpus}

def args_to_command(args):
    command = LSF_COMMAND + "\""
    command += PYTHON_COMMAND
    for key, value in args.items():
        command += f" --{key} {value}"
    command += "\""
    return command

def objective(config):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    config["checkpoint"] = checkpoint
    args = SimpleNamespace(**config)
    acc_vl, acc_te = run(args)
    train.report({"accuracy": acc_vl, "accuracy_te": acc_te})

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S") + "_" + args.dataset
    param_space = {
        "dataset": args.dataset,
        "hidden_features": tune.randint(32, 256),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-10, 1e-2),
        "length": tune.randint(3, 12),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": 1,
        "num_layers": 1, # tune.randint(1, 3),
        "num_samples": 8,
        "n_epochs": 50,  
        "patience": 5,
        "self_supervise_weight": tune.loguniform(1e-4, 1.0),
        "consistency_weight": tune.loguniform(1e-4, 1.0),
        "dropout": tune.uniform(0.0, 0.5),
        "checkpoint": 1,
        "activation": "SiLU", # tune.choice(["ReLU", "ELU", "SiLU"]),
        "batch_size": 1024,
    }

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=50,
        grace_period=5,
        reduction_factor=3,
        brackets=1,
    )

    tune_config = tune.TuneConfig(
        scheduler=scheduler,
        search_alg=OptunaSearch(),
        num_samples=1000,
        mode='max',
        metric='acc_vl',
    )

    storage_path = os.path.join(os.getcwd(), args.dataset)
    
    run_config = air.RunConfig(
        name=name,
        storage_path=storage_path,
        verbose=1,
    )

    if not args.restore_path:
        tuner = tune.Tuner(
            objective,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner.restore(args.restore_path,
                                   trainable=objective,
                                   param_space=param_space)

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RedditDataset")
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--restore_path", type=str, default=None) #'/home/user_name/NonConvGNN/Dataset/folder_name/
    args = parser.parse_args()
    experiment(args)
