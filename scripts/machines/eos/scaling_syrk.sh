#!/bin/bash
set -x


nodes=( 1 2 4 8 16 32 64 )
npats=( 487424 688128 974848 1376256 1949696 2752512 3506176 )


count=0
for node in "${nodes[@]}"
do
   submit_file=run_syrk_${node}nodes.sh
   npat=${npats[$count]}
   echo $npat
   cp run_syrk_template.sh ${submit_file}
   sed  -i "s,#NODES#,"${node}"," ${submit_file}
   sed  -i "s,#NPAT#,"${npat}"," ${submit_file}
   count=$((count+1))

   sbatch ${submit_file}
 
done
