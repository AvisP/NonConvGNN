import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from ray.tune.search import Repeater
import torch
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    ray.init(num_cpus=num_gpus, num_gpus=num_gpus)
    resource = {"cpu": num_gpus, "gpu": num_gpus}
else:
    import multiprocessing
    num_cpus = multiprocessing.count_cpus()
    ray.init(num_cpus=num_cpus)
    resource = {"cpu": num_cpus}

print(num_gpus)

def objective(config):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    config["checkpoint"] = checkpoint
    args = SimpleNamespace(**config)
    run(args)

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

    tuner = tune.Tuner(
        tune.with_resources(objective, resource),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="RedditDataset")
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--split_index", type=int, default=-1)
    args = parser.parse_args()
    experiment(args)
