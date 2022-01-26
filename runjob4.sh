#!/bin/bash
#SBATCH --job-name=mcm4
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCGibbs-remotesensing
#SBATCH --output=runjob4.out
#SBATCH --error=runjob4.err
#SBATCH --exclusive
 
python runFile_job4.py