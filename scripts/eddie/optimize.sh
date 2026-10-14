#!/bin/sh        
#$ -cwd                  
#$ -l h_rt=24:00:00 
#$ -l rl9=true

source /exports/csce/eddie/ph/groups/nnpdf/Users/ac/miniconda3/bin/activate map
export LD_LIBRARY_PATH=/exports/csce/eddie/ph/groups/nnpdf/Users/ac/Programs/apfelxx/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/exports/csce/eddie/ph/groups/nnpdf/Users/ac/Programs/NangaParbat/lib:$LD_LIBRARY_PATH
runcard=$1
fit_folder=$2
data=/exports/csce/eddie/ph/groups/nnpdf/Users/ac/codes/MontBlanc/data
optimize_path=/exports/csce/eddie/ph/groups/nnpdf/Users/ac/codes/MontBlanc/build/run/Optimize

$optimize_path $SGE_TASK_ID $runcard $data $fit_folder


