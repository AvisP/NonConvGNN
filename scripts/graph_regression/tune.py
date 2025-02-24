import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.search.ax import AxSearch
from ray.tune.search import Repeater
import torch
num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    resource = {"cpu": 6, "gpu": num_gpus}
else:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus)
    resource = {"cpu": num_cpus}

def objective(config):
    checkpoint = os.path.join(os.getcwd(), "model.pt")
    config["checkpoint"] = checkpoint
    args = SimpleNamespace(**config)
    rmse_vl, rmse_te = run(args)
    print('Rmse_vl ', rmse_vl, 'Rmse_te', rmse_te)
    train.report(dict(rmse_vl=rmse_vl, rmse_te=rmse_te))

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
        "num_layers": tune.randint(1, 3),
        "num_samples": 8,
        "n_epochs": 100,  
        "patience": 10,
        "self_supervise_weight": tune.loguniform(1e-4, 1.0),
        "consistency_weight": tune.loguniform(1e-4, 1.0),
        "dropout": tune.uniform(0.0, 0.5),
        "batch_size": tune.randint(32, 128),
        "checkpoint": 1,
        "activation": "SiLU", # tune.choice(["ReLU", "ELU", "SiLU"]),
        "split_index": args.split_index,
    }

    tune_config = tune.TuneConfig(
        metric="rmse_vl",
        mode="min",
        search_alg=Repeater(AxSearch(), 3),
        num_samples=2000,
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
            tune.with_resources(objective, {"cpu": args.use_cpu_per_trial, "gpu": args.use_gpu_per_trial}),
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner.restore(args.restore_path,
                                   trainable=tune.with_resources(objective, {"cpu": args.use_cpu_per_trial, "gpu": args.use_gpu_per_trial}),
                                   param_space=param_space)

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ESOL")
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--restore_path", type=str, default=None) #'/home/user_name/NonConvGNN/Dataset/folder_name/
    parser.add_argument("--use_cpu_per_trial", type=int, default=1)
    parser.add_argument("--use_gpu_per_trial", type=int, default=0)
    args = parser.parse_args()
    if num_cpus < args.use_cpu_per_trial:
        print(f"WARNING : Ray intialized with {num_cpus} cpus Cannot allocate {args.use_cpu_per_trial} cpus. Training will FAIL even if it starts!!")
    if num_gpus < args.use_gpu_per_trial:
        print(f"WARNING : Ray intialized with {num_gpus} gpus Cannot allocate {args.use_gpu_per_trial} gpus. Training will FAIL even it starts!!")
    experiment(args)
