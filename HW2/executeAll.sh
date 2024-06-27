#!/usr/bin/env bash

for execs in {2,4,8,16};
do	for i in `seq 1 3`;
	do echo $execs executors
	spark-submit --num-executors $execs G007HW2.py /data/BDC2324/artificial100M_9_100.csv 10 110 16
	done
done

for K in {50,70,90,110,130};
do echo $K centers
	spark-submit --num-executors 16 ./G007HW2.py /data/BDC2324/artificial10M_9_100.csv 3 $K 16
done
