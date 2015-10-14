#!/bin/bash
#PBS -N axpy
#PBS -lwalltime=00:05:00
#PBS -lnodes=1:ppn=6

cd $PBS_O_WORKDIR

nrep=25
L=1024
while [ $L -le $((1024*1024*4)) ] ; do
    ./axpy $L $nrep
    let L=L*2
done | tee axpy.log
