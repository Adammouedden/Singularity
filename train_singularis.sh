#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH -n8
#SBATCH --constraint=gpu80
#SBATCH -p highgpu
#SBATCH --job-name=singularis-urm-training
#SBATCH --mem=64G
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/cuda-12.x.x        # replace with whatever cuda/12.x module name your cluster has
                                    # run `module avail cuda` to check
module list
nvidia-smi topo -m
nvidia-smi

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"

mkdir -p logs checkpoints

# ── Run ───────────────────────────────────────────────────────────────────────
cd /lustre/fs1/home//home/ad028676/RESEARCH/Singularity
uv run python -m reasoning.singularis.training.train_Singularis

echo "Done: $(date)"