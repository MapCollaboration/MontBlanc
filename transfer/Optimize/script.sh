#!/bin/bash
#SBATCH -p standard
#SBATCH --chdir=/home/lordguto/Montblanc/build/run

## generation of Optimize command one for each replica

jbegin=1
nrep=250
cpus=6
jend= $cpus
mode=${3:-'f1D1'}
for i in #insert nodenames 
 do
    for (( r=$jbegin; r <= $jend; r++ ))
     do
        occam-run -n $i -s something Optimize ./Optimize $r ../../config/MAPFF20/PI/MAPFF20_PI_NNLO_Q1_00.yaml ../../data/ fit/
        if [$r -eq $nrep]
        then
          break
        fi
     done
     if [$r -eq $nrep]
     then
        break
     fi
     jbegin=$(($jbegin + $cpus))
     jend=$(($jend+ $cpus))
 done
