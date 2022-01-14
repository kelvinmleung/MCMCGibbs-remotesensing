#!/bin/bash
#SBATCH --job-name=mcm1
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCGibbs-remotesensing
#SBATCH --output=runjob1.out
#SBATCH --error=runjob1.err
#SBATCH --exclusive
 
python runFile_job1.py