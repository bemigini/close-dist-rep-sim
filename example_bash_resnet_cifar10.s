#!/bin/bash	

#SBATCH --job-name=resnetcifar10_var
#SBATCH --output=resnetcifar10_var_result-%J.out
#SBATCH --cpus-per-task=1
#SBATCH --time=95:00:00
#SBATCH --mem=14gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=user@mail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"


conda init bash
source ~/.bashrc

source /opt/miniconda3/etc/profile.d/conda.sh

cd close-dist-rep-sim/ 
conda activate close-dist

echo -e "Working dir: $(pwd)\n"

python run.py train-variations --output-folder=checkpoints --dataset-config=configs/cifar10_0_cls10.json --model-var-config=configs/model_variations_config_resnetcifar10_128_fd3.json --train-config=configs/cifar10_0_32_ADAM_0_0001_20000steps.json --date-str=2025-04-22 --cuda


echo "Done: $(date +%F-%R:%S)"
