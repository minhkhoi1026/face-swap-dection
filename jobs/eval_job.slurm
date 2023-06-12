#!/bin/bash
#SBATCH --job-name=eval-model    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=minhkhoi1026@gmail.com
#SBATCH --nodelist=selab4

module purge

# module load graphviz-2.49.0-gcc-9.3.0-nc4zl4i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate fsd-torch

pip install .

python src/test.py -c configs/single_head.yml
