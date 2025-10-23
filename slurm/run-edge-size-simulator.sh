#!/usr/bin/env bash

#SBATCH --job-name=toy-edge-size-simulator             
#SBATCH --array=0-20                     
#SBATCH --partition=standard
#SBATCH --time=24:00:00                        
#SBATCH --output=throughput/logs/edge-sizes/toy_edge_size_simulator_%a.out
#SBATCH --mem-per-cpu=20gb


# print SLURM environment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" 
echo "Starting: "`date +"%D %T"` 
# perform compute
    
env/bin/python -u -m scripts.toy_edge_size_simulator $SLURM_ARRAY_TASK_ID

# done

# End of job info 
echo "Ending: "`date +"%D %T"`