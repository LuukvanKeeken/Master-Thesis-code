#!/bin/bash
jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.01lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_0.0005lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_Standard_RNN_96neurons_5e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.01lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_0.0005lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_1e-05lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.0001ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.0001ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.0001ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.0001ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.01ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.01ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.01ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_0.01ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_1.0ec_0.0001vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_1.0ec_0.01vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_1.0ec_0.5vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
while [ $(squeue -u $USER | wc -l) -gt 120 ]; do
    sleep 60
done

jobid=$(sbatch $HOME/Thesis/29_BW_BPandRNN_original_96/jobscript_BP_RNN_96neurons_5e-06lr_1.0ec_1.0vc.sh | awk '{print $4}')
echo $jobid >> jobids_29_BW_BPandRNN_original_96.txt
