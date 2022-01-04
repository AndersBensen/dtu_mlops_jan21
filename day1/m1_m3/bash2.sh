#!/bin/bash
# A bash script by bo

for ((i=0; i<5; i++))
do
	echo "Called $i times"
	python3 helloworld.py
done

