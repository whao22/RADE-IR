#!/bin/bash
#SBATCH --job-name=3dgsa-ir
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --partition=L40
#SBATCH --gres=gpu:1

# show currrent status
echo Start time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
# pip install gpustat
gpustat


# for crontab
cd `echo $PWD`

################################################################
##################### CUSTOM SCRIPTS START #####################

# # SyntheticHuman
# python train_manual.py dataset=jody option=iter15k
# python train_manual.py dataset=megan option=iter15k
# python train_manual.py dataset=ps_female_3 option=iter50k
# python train_manual.py dataset=ps_female_4 option=iter50k

# # ZJU-MoCap
# python train_manual.py dataset=zjumocap_393_mono option=iter50k
# python train_manual.py dataset=zjumocap_394_mono option=iter50k
# python train_manual.py dataset=zjumocap_386_mono option=iter50k
# python train_manual.py dataset=zjumocap_387_mono option=iter50k
# python train_manual.py dataset=zjumocap_377_mono option=iter50k

# People-Snapshot
python train_manual.py dataset=ps_male_3 option=iter30k
python train_manual.py dataset=ps_male_4 option=iter30k
python train_manual.py dataset=ps_female_3 option=iter30k
python train_manual.py dataset=ps_female_4 option=iter30k


###################### CUSTOM SCRIPTS END ######################
################################################################
echo Current time is `date`
echo Job completed!
