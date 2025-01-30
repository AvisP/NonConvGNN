# Random Walk with Unifying memory

This repo cotains an implementation of the paper [Non-Convolutional Graph Neural Network](https://arxiv.org/abs/2408.00165). The code has been restructured from the original [repo](https://github.com/yuanqing-wang/rum) and some modifications done to deal with package dependency and import issues that was arsing with original one.

All folders have a `run.py` and a `tuner.py` which passes different hypter parameter as arguments to `run.py` to fir the best model. The Random Walk with Unifying Memory clss is in the `rum` folder. The different folders correpond to diferent datasets and tasks i.e. graph classification/regression and node classification on different datasets. Detailed explanation is provided below. The scripts can be run from the main folder by `pyhton scripts/tu/tuner.py`.

**Graph Classification**

| Dataset name  | Folder Name  |
| ------------- | ------------- |
| MUTAG         | tu/tune.py  |
| IMDB          | tu/tune_multi.py  |
| CLLAB         | tu/tune_multi.sbatch |
| NCI1          | tu/tune.sbatch |
