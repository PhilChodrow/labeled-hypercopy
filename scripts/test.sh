#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=slurm_test      # Job name
#SBATCH --output=slurm_test-%j.out # Output file incorporating job ID
#SBATCH --partition=standard        # Partition (queue) 
#SBATCH --array=1-10
#SBATCH --time=00:05:00             # Time limit hrs:min:sec
#SBATCH --mem=500mb                 # Job memory request 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fcataldo@middlebury.edu


# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here

echo $PWD

.venv/bin/python slurm_test.py -${SLURM_JOB_ID}


# End of job info 
echo "Ending: "`date +"%D %T"`

