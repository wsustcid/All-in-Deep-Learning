#!/bin/bash

#PBS -N xu_hill              # Job name
#PBS -l nodes=1:gpus=1:S       # Allocate two GPUs
#PBS -o job_$PBS_JOBID.out      # stdout
#PBS -e job_$PBS_JOBID.err      # stderr

echo This job runs on following nodes:
cat $PBS_NODEFILE
echo Allocated GPUs:
cat $PBS_GPUFILE
echo ===== OUTPUT =====
startdocker -u "-v /gdata/$USER:/gdata/$USER -w /gdata/$USER" -c 'python /gdata/chenkj/main.py' bit:5000/deepo

