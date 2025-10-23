#!/usr/bin/env bash

#SBATCH --job-name=toy-edge-size-simulator             
#SBATCH --mem=2gb                              
#SBATCH --partition=standard
#SBATCH --time=24:00:00                        
#SBATCH --output=throughput/logs/edge-sizes/long-run.out

# print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 
echo "Starting: "`date +"%D %T"` 
# perform compute
    
env/bin/python -u -m scripts.stationarity-analysis 

# done

# End of job info 
echo "Ending: "`date +"%D %T"`