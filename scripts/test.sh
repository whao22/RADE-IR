#!/bin/bash
#SBATCH --job-name=3dgsa-ir
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=128G
#SBATCH --partition=gpujl
#SBATCH --gres=gpu:1

# show currrent status
echo Start time is `date`
echo Directory is $PWD 
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
conda info --envs
gpustat

# for crontab
cd `echo $PWD`

################################################################
##################### CUSTOM SCRIPTS START #####################

# TEST
python render.py mode=test dataset.test_mode=pose dataset=ps_female_4 hdr=city option=iter30k evaluate=true
# python render.py mode=test dataset.test_mode=pose dataset=ps_male_3 hdr=bridge option=iter50k evaluate=false
# python render.py mode=test dataset.test_mode=pose dataset=ps_female_4 option=iter15k
# python render.py mode=test dataset.test_mode=pose dataset=ps_female_4 hdr=bridge option=iter15k

###################### CUSTOM SCRIPTS END ######################
################################################################
echo Current time is `date`
echo Job completed!
