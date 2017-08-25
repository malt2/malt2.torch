#!/bin/bash
#Placement script for running mpi jobs over multiple GPUs
#NEC Labs, 2017
echo -e "\xF0\x9F\x8D\xBB MALT Torch running over multiple GPUs in round-robin fashion"
prog=$@
echo 'Program to run is :' $prog
lrank=$OMPI_COMM_WORLD_LOCAL_RANK
ngpus=$(ls /proc/driver/nvidia/gpus | wc -l)
echo ' Found total GPUS :' $ngpus 
gputouse=$(($lrank % $ngpus))
echo ' Running on GPU : ' $gputouse
#gputouse=0
#othergpu=$((($gputouse-1)*($gputouse-1)))
for i in `seq 0 $(($ngpus-1))`
do
    if [ $i -ne $gputouse ]
    then
        othergpu=$othergpu","$i
    fi
done
echo ' Other GPU is : ' $othergpu
echo ' Run command is:  export CUDA_VISIBLE_DEVICES='$gputouse$othergpu';' $prog';'
export CUDA_VISIBLE_DEVICES=$gputouse$othergpu; $prog;
