#!/bin/bash
#SBATCH --job-name=mcm2
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCGibbs-remotesensing
#SBATCH --output=runjob2.out
#SBATCH --error=runjob2.err
#SBATCH --exclusive
 
python runFile_job1.py