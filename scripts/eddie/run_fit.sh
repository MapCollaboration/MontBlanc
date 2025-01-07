#!/bin/sh

# Usage:
# > bash run_fit.sh <path to runcard> <nrep> <path to fit fodler>

source /exports/csce/eddie/ph/groups/nnpdf/Users/ac/miniconda3/bin/activate map

mkdir -p logs

runcard=$1
nrep=$2
fit_folder=$3

runcard_name=$(basename "$runcard")

qsubpath=/exports/applications/gridengine/ge-8.6.5/bin/lx-amd64/qsub

FIT_LOG_FOLDER=logs/logs_${runcard_name%".yml"}
FIT_LOG_FOLDER=$FIT_LOG_FOLDER/fit
mkdir -p $FIT_LOG_FOLDER

$qsubpath -N "MontBlanc_"${runcard_name%".yml"} -e $FIT_LOG_FOLDER -o $FIT_LOG_FOLDER -t 1-$nrep -l h_vmem=2500M -pe sharedmem 4 optimize.sh $runcard $fit_folder

