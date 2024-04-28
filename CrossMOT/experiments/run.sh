#!/bin/bash
#SBATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-01:00:00
#SBATCH --mail-user=harshit.aggarwal@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -o /home2/harshit.aggarwal/DIVOTrack/logs/divotrack_ablations_with_conflict_free.out # To change
#SBATCH -c 27 # To change
#SBATCH --gres=gpu:3 # To change

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CrossMOT

echo "Activated"

cd ~/DIVOTrack/CrossMOT/src

python train.py mot --exp_id DIVOTrack_ablations_with_confict_free --data_cfg '../src/lib/cfg/divo.json' --load_model "../models/fairmot_dla34.pth" --gpus 0,1,2 --batch_size 8 --num_epochs 30 --single_view_id_split_loss
# Command To change

echo "Done"
