#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=senate_bills      # Job name
#SBATCH --output=senate_bills%j.out # Output file incorporating job ID
#SBATCH --partition=long        # Partition (queue) 
#SBATCH --time=100:00:00             # Time limit hrs:min:sec
#SBATCH --mem=128G                 # Job memory request 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fcataldo@middlebury.edu


# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here
.venv/bin/python src/senate_bills_algo_test.py


# End of job info 
echo "Ending: "`date +"%D %T"`

