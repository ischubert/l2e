#!/bin/bash

if [ -z "$1" ]; then
    echo "ERROR: CONFIG_ID was not specified"
    exit
fi

## Get config file and load env variables from json
export CONFIG_ID=$1
export CONFIG_FILE=config_$CONFIG_ID.json
for s in $(cat $CONFIG_FILE | jq -r "to_entries|map(\"\(.key)=\(.value|tostring)\")|.[]"); do
    export $s
done

for SLURM_ARRAY_TASK_ID in $(seq $AGENTS_MIN $AGENTS_MAX)
do
    echo Deploying agent $SLURM_ARRAY_TASK_ID
    export SLURM_ARRAY_TASK_ID
    LEAST_BUSY_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,nounits,noheader | sort -nk2 | awk -F "," 'NR==1{print $1}')
    OUTFILE=${scratch_root}logs/$CONFIG_ID.$SLURM_ARRAY_TASK_ID.out
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$LEAST_BUSY_GPU screen -dmS $CONFIG_ID.$SLURM_ARRAY_TASK_ID -L -Logfile $OUTFILE python $SCRIPT_ID.py
    sleep 15
done