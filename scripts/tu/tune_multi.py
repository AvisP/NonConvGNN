import os
from types import SimpleNamespace
from datetime import datetime
from run_multi import run
import ray
from ray import tune, air, train
# from ray.tune.trainable import session
from ray.tune.search.ax import AxSearch
# from ray.tune.search import Repeater
import torch
num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    resource = {"cpu": num_gpus, "gpu": num_gpus}
else:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus)
    resource = {"cpu": num_cpus}

def objective(config):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    config["checkpoint"] = checkpoint
    args = SimpleNamespace(**config)
    acc, acc_std = run(args)
    # train.report(dict(acc=acc, acc_std=acc_std))
    train.report({"acc": acc, "acc_std": acc_std})

def experiment(args):
    name = datetime.now().strftime("%m%d%Y%H%M%S") + "_" + args.data
    param_space = {
        "data": args.data,
        "hidden_features": tune.randint(32, 64),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-8, 1e-2),
        "length": tune.randint(3, 16),
        "consistency_temperature": tune.uniform(0.0, 1.0),
        "optimizer": "Adam",
        "depth": 1,
        "num_layers": 1, # tune.randint(1, 3),
        "num_samples": 4,
        "n_epochs": 500,  
        "patience": tune.randint(5, 20),
        "factor": tune.uniform(0.1, 0.8),
        "self_supervise_weight": tune.loguniform(1e-5, 1.0),
        "consistency_weight": tune.loguniform(1e-5, 1.0),
        "dropout": tune.uniform(0.0, 1.0),
        "batch_size": tune.randint(32, 128),
        "checkpoint": 1,
        "activation": "SiLU", # tune.choice(["ReLU", "ELU", "SiLU"]),
        "split_index": args.split_index,
    }

    tune_config = tune.TuneConfig(
        metric="acc",
        mode="max",
        search_alg=AxSearch(),
        num_samples=1000,
    )

    if args.split_index < 0:
        storage_path = os.path.join(os.getcwd(), args.data)
    else:
        storage_path = os.path.join(os.getcwd(), args.data, str(args.split_index))
    
    run_config = air.RunConfig(
        name=name,
        storage_path=storage_path,
        verbose=0,
    )

    if not args.restore_path:
        tuner = tune.Tuner(
            tune.with_resources(objective, {"cpu": 1, "gpu": 1}),
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner.restore(path=args.restore_path,
                               trainable=objective,
                               param_space=param_space)

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="MUTAG")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--restore_path", type=str, default=None)
    args = parser.parse_args()
    experiment(args)
