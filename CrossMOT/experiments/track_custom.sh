BATCH -A research
#SBATCH --partition=long
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-01:00:00
#SBATCH --mail-user=bhav.beri@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH -o /home2/maharnav.singhal/DIVOTrack/logs/track_with_conflict_free.out # To change
#SBATCH -c 9 # To change
#SBATCH --gres=gpu:1 # To change

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CrossMOT

pip install scikit-learn

echo "Activated"

cd ~/DIVOTrack/CrossMOT/src

model_path="/home2/maharnav.singhal/DIVOTrack/CrossMOT/exp/mot/DIVOTrack_ablations_with_confict_free1/model_last.pth"
exp_name="track_with_conflict_free"

python track.py mot --load_model $model_path --test_divo --conf_thres 0.5 --reid_dim 512 --exp_name $exp_name --exp_id $exp_name

