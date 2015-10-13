#!/bin/bash
#PBS -N mxam
#PBS -lwalltime=00:15:00
#PBS -lnodes=1:ppn=12

module unload parastation
module load parastation/gcc

cd $PBS_O_WORKDIR
nrep=15
L=$((512*1024))

LOGFILE=mxam.log
rm -f $LOGFILE
for nthr in 12 6 4; do
    echo '# nthr = ' $nthr >> $LOGFILE
    for N in $(seq 4 16) ; do
	make clean
	N=$N make
	export OMP_NUM_THREADS=$nthr
	./mxam $L $nrep >> $LOGFILE
    done
    echo >> $LOGFILE
    echo >> $LOGFILE
done
