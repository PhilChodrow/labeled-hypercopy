#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=best_algo_on_senate_bills     # Job name
#SBATCH --output=best_algo_on_senate_bills-%j.out # Output file incorporating job ID
#SBATCH --partition=standard        # Partition (queue) 
#SBATCH --time=20:00:00             # Time limit hrs:min:sec
#SBATCH --mem=8G                 # Job memory request 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fcataldo@middlebury.edu


# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here

echo $PWD

.venv/bin/python senate_bills_algo_test.py -4


# End of job info 
echo "Ending: "`date +"%D %T"`

