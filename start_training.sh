#!/bin/bash
cd /home/icg130/Schreibtisch/deeptrac-mini
nohup python training.py > training_output.log 2>&1 &
echo "Training started. PID: $!"
echo "Output: training_output.log"
echo "Log: training.log"
