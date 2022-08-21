#!/bin/bash
#SBATCH --job-name=eval-model    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=minhkhoi1026@gmail.com

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate fsd 

python eval.py --bs 20 --dim 128 --backbone Xception --num-classes 1 --weight weights/weight-20220820-124207-Xception-09-0.90-0.82-0.40303.hdf5
