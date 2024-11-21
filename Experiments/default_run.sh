#!/bin/bash
#SBATCH --job-name=defPTM
#SBATCH -p gpu
#SBATCH --gpus=1

#            d-hh:mm:ss
#SBATCH --time=00:30:00
module load 2023 
module load CUDA/12.1.1 
module load cuDNN/8.9.2.26-CUDA-12.1.1 


# Environment (Snellius specific)
source /sw/arch/Centos8/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh
#conda env create -f ../environment.yml --name dgl_ptm_gpu
conda activate dgl_ptm_gpu

# Experimental setup

# Read seeds from seeds.txt
seeds=()
readarray -t seeds < <(cat seeds.txt | tr ',' '\n' | tr -s ' ' '\n')

total_runs=${#seeds[@]}
counter=0
restart=0
earlystop=50

echo "Script: default_run.sh"

for seed in "${seeds[@]}"
    do
        ((counter++))
        if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
            date=$(date)
            echo "$date Started run $counter/$total_runs with seed: $seed"
            variation="--seed $seed --steps 75"
            python gpu_default.py $variation 
            date=$(date)
            echo "$date Finished run $counter/$total_runs with seed: $seed"
        fi
    done


wait


echo " $date - Runs $restart through $earlystop have been attempted."

