#!/bin/bash
#SBATCH --job-name=extract-train    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=minhkhoi1026@gmail.com

module purge

source ~/miniconda3/etc/profile.d/conda.sh

conda activate fsd

python scripts/extract_frames.py --source ./data/deepfaker_app/ --dest ./data_extract/deepfaker_app/face_crop/ --sampling-ratio 1 --extract-type face
python scripts/extract_landmark.py --source data_extract/deepfaker_app/face_crop/ --dest data_extract/deepfaker_app/landmark_extract/