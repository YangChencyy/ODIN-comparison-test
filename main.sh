#!/bin/bash

#SBATCH --account=sunwbgt98
#SBATCH --job-name=ODIN
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --mail-user=rivachen@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=standard
#SBATCH --output=/home/rivachen/ODIN-comparison-test/results.log

module purge
conda init bash
conda activate GP

python ODIN.py 'MNIST' 128 128 0