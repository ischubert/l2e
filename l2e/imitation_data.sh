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

for SLURM_ARRAY_TASK_ID in $(seq 1 $n_data_collect_workers)
do
    echo Deploying agent $SLURM_ARRAY_TASK_ID
    export SLURM_ARRAY_TASK_ID
    OUTFILE=${scratch_root}logs/$DATA_COLLECTION_SCRIPT_ID.$CONFIG_ID.$SLURM_ARRAY_TASK_ID.out
    num_cpus=$(getconf _NPROCESSORS_ONLN)
    cpu_now=$(echo $(($RANDOM % $num_cpus)))
    CUDA_VISIBLE_DEVICES=-1 taskset -c $cpu_now screen -dmS $DATA_COLLECTION_SCRIPT_ID.$CONFIG_ID.$SLURM_ARRAY_TASK_ID -L -Logfile $OUTFILE python $DATA_COLLECTION_SCRIPT_ID.py
done