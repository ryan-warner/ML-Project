#!/bin/bash
#SBATCH -J MLProjectJob                         # Job name
#SBATCH --gres=gpu:V100:1                        # GPU Type and Count
#SBATCH -N1                                     # Number of nodes and cores per node required
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=1G                        # Memory per core
#SBATCH --time 2:00:00                          # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=rwarner31@gatech.edu        # E-mail address for notifications
#SBATCH --partition=coc-gpu
#SBATCH --tmp=32G
cd $SLURM_SUBMIT_DIR      

module load anaconda3
conda activate ML-Project
srun python main.py
