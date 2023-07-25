#!/bin/bash
#SBATCH --job-name=eval-model    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --nodelist=selab2

module purge

# module load graphviz-2.49.0-gcc-9.3.0-nc4zl4i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate fsd

pip install .

python src/test.py -c configs/train/double_head_vit_msr_hybrid.yml
# python src/test.py -c configs/train/double_head_vit_fafi_hybrid.yml
# python src/test.py -c configs/train/double_head_mobilenet_msr_hybrid.yml
# python src/test.py -c configs/train/double_head_mobilenet_fafi_hybrid.yml
python src/test.py -c configs/train/double_head_restnet_msr_hybrid.yml
python src/test.py -c configs/train/double_head_restnet_fafi_hybrid.yml
