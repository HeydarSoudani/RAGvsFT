#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --output=script_logging/slurm_%A.out

# Loading modules
module load 2022
# module load foss/2022a
module load Python/3.10.4-GCCcore-11.3.0

# pip install -r $HOME/RAGvsFT/component0_preprocessing/requirements.txt
srun $HOME/RAGvsFT/component0_preprocessing/create_costomized_dataset.py \
    --qg_model 'lmqg/t5-large-squad-qg' \
    --ae_model 'lmqg/t5-large-squad-ae'

