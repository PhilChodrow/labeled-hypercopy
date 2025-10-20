#!/usr/bin/env bash

# SLURM template for serial jobs

# Set SLURM options
#SBATCH --job-name=parameter_sweep      # Job name
#SBATCH --output=parameter_sweep%j.out # Output file incorporating job ID
#SBATCH --partition=standard        # Partition (queue) 
#SBATCH --array=0-120
#SBATCH --time=10:00:00             # Time limit hrs:min:sec
#SBATCH --mem=8G                 # Job memory request 
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=fcataldo@middlebury.edu


# Print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 

# Start of job info
echo "Starting: "`date +"%D %T"` 

# Your calculations here
.venv/bin/python parameter_sweep.py -${SLURM_ARRAY_TASK_ID}


# End of job info 
echo "Ending: "`date +"%D %T"`

