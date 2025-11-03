#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=timing      # Job name
#SBATCH --output=timing%j.out # Output file incorporating job ID
#SBATCH --partition=long        # Partition (queue) 
#SBATCH --time=100:00:00             # Time limit hrs:min:sec
#SBATCH --mem=8G                 # Job memory request 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fcataldo@middlebury.edu


# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here
.venv/bin/python src/simulated_annealing_timer.py


# End of job info 
echo "Ending: "`date +"%D %T"`

