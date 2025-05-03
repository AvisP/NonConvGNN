import os
from types import SimpleNamespace
from datetime import datetime
from run import run
import ray
from ray import tune, air, train
from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search import Repeater
import torch
num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
    resource = {"cpu": num_cpus, "gpu": num_gpus}
else:
    num_cpus = os.cpu_count()
    ray.init(num_cpus=num_cpus)
    resource = {"cpu": num_cpus}

def objective(config):
    global checkpoint_dir

    trial_dir = tune.get_context().get_trial_dir()
    
    args = SimpleNamespace(**config)
    acc_vl, acc_te, model = run(args)
    print('Acc_vl ', acc_vl, 'Acc_te ', acc_te)

    if os.listdir(checkpoint_dir):  # Check if the folder contains files
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            if os.path.isfile(file_path):  # Ensure it's a file, not a folder
                os.remove(file_path)
                print(f"Removed: {file}")
        print("All files have been removed!")
    else:
        print("Folder is empty.")

    checkpoint_path = os.path.join(checkpoint_dir, f"model_{trial_dir.split('/')[-1].split('_')[1]}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print("SAVED AT ", checkpoint_path)
    checkpoint = train.Checkpoint.from_directory(checkpoint_dir)
    tune.report(metrics=dict(acc_vl=acc_vl, acc_te=acc_te), checkpoint=checkpoint)

def experiment(args):
    global checkpoint_dir
    name = datetime.now().strftime("%m%d%Y%H%M%S") + "_" + args.data
    checkpoint_dir = os.path.join(os.getcwd(), f"{args.data}", name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        "n_epochs": 500,  
        "patience": 500,
        "self_supervise_weight": tune.loguniform(1e-4, 1.0),
        "consistency_weight": tune.loguniform(1e-4, 1.0),
        "dropout": tune.uniform(0.0, 0.5),
        "checkpoint": 1,
        "activation": "SiLU", # tune.choice(["ReLU", "ELU", "SiLU"]),
        "split_index": args.split_index,
        "directed": args.directed,
        "seed": 3615
    }

    tune_config = tune.TuneConfig(
        metric="acc_vl",
        mode="max",
        search_alg=OptunaSearch(),
        num_samples=4,
    )

    if args.split_index < 0:
        storage_path = os.path.join(os.getcwd(), args.data)
    else:
        storage_path = os.path.join(os.getcwd(), args.data, str(args.split_index))
    
    run_config = tune.RunConfig(
        name=name,
        storage_path=storage_path,
        verbose=0,
        checkpoint_config=tune.CheckpointConfig(
            num_to_keep=3,  # Keep last 5 checkpoints
            checkpoint_score_attribute='acc_vl',
            checkpoint_score_order='max'),     
    )

    if not args.restore_path:
        tuner = tune.Tuner(
            tune.with_resources(objective, {"cpu": args.use_cpu_per_trial, "gpu": args.use_gpu_per_trial}),
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    else:
        tuner = tune.Tuner.restore(path=args.restore_path,
                               trainable=tune.with_resources(objective, {"cpu": args.use_cpu_per_trial, "gpu": args.use_gpu_per_trial}),
                               param_space=param_space)

    results = tuner.fit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="AmazonCoBuyComputerDataset")
    parser.add_argument("--directed", type=int, default=0)
    parser.add_argument("--split_index", type=int, default=-1)
    parser.add_argument("--restore_path", type=str, default=None)
    parser.add_argument("--use_cpu_per_trial", type=int, default=1)
    parser.add_argument("--use_gpu_per_trial", type=int, default=0)
    args = parser.parse_args()
    if num_cpus < args.use_cpu_per_trial:
        print(f"WARNING : Ray intialized with {num_cpus} cpus Cannot allocate {args.use_cpu_per_trial} cpus. Training will FAIL even if it starts!!")
    if num_gpus < args.use_gpu_per_trial:
        print(f"WARNING : Ray intialized with {num_gpus} gpus Cannot allocate {args.use_gpu_per_trial} gpus. Training will FAIL even it starts!!")
    experiment(args)
