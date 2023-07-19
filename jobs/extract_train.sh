#!/bin/bash
#SBATCH --job-name=extract-train    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)


module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate fsd

python scripts/extract_all.py  --source data/swapface --dest data_extract/swapface --sampling-ratio 1
