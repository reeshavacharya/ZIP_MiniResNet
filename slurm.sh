#!/bin/bash -l

#SBATCH --job-name=zip-table5
# Use the Quick partition (GPU1 is only in Quick)
#SBATCH --partition=Quick

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
# Quick partition has 1-day limit
#SBATCH --time=1-00:00:00

# Pin job to node GPU1 and request one A100 GPU
# (this node has Gres=gpu:A100:3)
#SBATCH --nodelist=GPU1
#SBATCH --gres=gpu:A100:1

#SBATCH --output=std_out
#SBATCH --error=std_err

GOTMPDIR=~/go-tmp TMPDIR=~/go-tmp
srun scripts/table_precomputed.sh
