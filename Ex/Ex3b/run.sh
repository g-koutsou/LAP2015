#!/bin/bash
#PBS -N lapl 
#PBS -lwalltime=00:05:00
#PBS -lnodes=1:ppn=6

cd $PBS_O_WORKDIR

L=$((4*1024))
./lapl $L |tee lapl.log
