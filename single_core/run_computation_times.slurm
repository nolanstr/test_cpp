#!/bin/bash
#SBATCH --time=120:00:00 # walltime, abbreviated by -t
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1079968@umail.utah.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# additional information for allocated clusters
#SBATCH --account=hochhalter-np     # account - abbreviated by -A
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p

##SBATCH --pty 
##SBATCH --qos=hochhalter-high-np

python test.py bingo_cpp

mkdir ~/codes/bingo/tmp
mv ~/codes/bingo/bingocpp ~/codes/bingo/tmp/.

python test.py bingo_no_cpp
mv ~/codes/bingo/tmp/bingocpp ~/codes/bingo/.
rm -rf ~/codes/bingo/tmp

