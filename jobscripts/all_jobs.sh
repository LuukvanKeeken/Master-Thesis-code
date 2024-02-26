#!/bin/bash
sbatch jobscript_32neurons_0.0001lr_0.2rf.sh
sleep 20
sbatch jobscript_32neurons_5e-05lr_0.2rf.sh
sleep 20
sbatch jobscript_32neurons_0.0001lr_0.35rf.sh
sleep 20
sbatch jobscript_32neurons_5e-05lr_0.35rf.sh
sleep 20
sbatch jobscript_32neurons_0.0001lr_0.5rf.sh
sleep 20
sbatch jobscript_32neurons_5e-05lr_0.5rf.sh
sleep 20
