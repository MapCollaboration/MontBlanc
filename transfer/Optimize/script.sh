#!/bin/bash
#SBATCH -p standard

##SBATCH --chdir=/home/lordguto/Montblanc/build/run

## generation of Optimize command one for each replica

jbegin=1
nrep=4
cpus=6
jend= $cpus
counter=0
mode=${3:-'f1D1'}
for (( i=1; i<= 4; i++)) #$i in #insert nodenames 
 do
    for (( r=$jbegin; r <= $jend; r++ ))
     do
        occam-run -s -v archive/home/lorenzo.canzian/volume:../../../volume lorenzo.canzian/optimize ./Optimize $r ../../../volume/test.yaml ../../data/ fit/
        counter=$(($counter + 1))
        if [$counter -eq $nrep ]
        then
          break
        fi
     done
     if [$counter -eq $nrep]
     then
        break
     fi
     jbegin=$(($jbegin + $cpus))
     jend=$(($jend+ $cpus))
 done
