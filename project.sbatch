#!/bin/bash
#
#SBATCH --job-name=prl_KM
#SBATCH --output=out.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

echo ""
gcc -fopenmp k-means0.c -o k-means0.o
./k-means0.o 

gcc -fopenmp k-means1.c -o k-means1.o
./k-means1.o 

gcc -fopenmp k-means2.c -o k-means2.o
./k-means2.o 

gcc -fopenmp k-means3.c -o k-means3.o
./k-means3.o 
