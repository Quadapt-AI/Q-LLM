import os
import logging
logging.basicConfig(format="", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
wflag = 0
set_flag = [True, False]
def mkdir_p(dir_p):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir_p):
        os.makedirs(dir_p)
    
path = os.getcwd()
job_directory = os.path.join(path, 'jobs')
# Make top level directories
mkdir_p(job_directory)
MODEL_CLASSES = {
                    'gpt2': 'gpt2-b',
                    'gpt2-medium': 'gpt2-m',
                    'gpt2-large': 'gpt2-l',
                    'gpt2-xl': 'gpt2-xl',
                    'openai-gpt': 'gpt3',
                    'facebook/bart-large-cnn': 'bart',
                    'bert-base-uncased': 'bert', 
                    'roberta-base': 'roberta', 
                    }
# vae_loss_typ = ['vae', 'betavae', 'controlvae', 'infovae', 'gcvae']
mmd_types = ['mmd', 'mah']
vae_loss_typ = ['vae', 'controlvae', 'gcvae']
exceptions = ['bert', 'roberta'] #EncoderDecoder Model


for i, j in MODEL_CLASSES.items():
    for vl in vae_loss_typ:
        if not vl == 'gcvae':
            job_file = os.path.join(job_directory, f"{i.replace('/', '_').replace('-', '_')}_{vl[:1]}.job")
            with open(job_file, 'w+') as writer:
                writer.writelines('#!/bin/bash -l\n')
                writer.writelines(f'#SBATCH --job-name={j[::-1]}_{vl[:1]}\n') #reverse model name for anonimity
                writer.writelines('#SBATCH --mail-user=ifeanyi.ezukwoke@emse.fr\n')
                writer.writelines('#SBATCH --mail-type=ALL\n')
                writer.writelines('#SBATCH --gres=gpu:1\n')
                writer.writelines('#SBATCH --nodes=1\n')
                writer.writelines('#SBATCH --ntasks=1\n')
                writer.writelines('#SBATCH --cpus-per-task=4\n')
                writer.writelines('#SBATCH --mem=50G\n')
                writer.writelines('#SBATCH --time=7-00:00:00\n')
                writer.writelines('#SBATCH --partition=audace2018\n')
                writer.writelines('ulimit -l unlimited\n')
                writer.writelines('unset SLURM_GTIDS\n')
                writer.writelines('echo -----------------------------------------------\n')
                writer.writelines('echo SLURM_NNODES: $SLURM_NNODES\n')
                writer.writelines('echo SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST\n')
                writer.writelines('echo SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR\n')
                writer.writelines('echo SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST\n')
                writer.writelines('echo SLURM_JOB_ID: $SLURM_JOB_ID\n')
                writer.writelines('echo SLURM_JOB_NAME: $SLURM_JOB_NAME\n')
                writer.writelines('echo SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION\n')
                writer.writelines('echo SLURM_NTASKS: $SLURM_NTASKS\n')
                writer.writelines('echo SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE\n')
                writer.writelines('echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE\n')
                writer.writelines('echo -----------------------------------------------\n')
                writer.writelines('echo Run program...\n')
                writer.writelines('source ~/meso-env/env.sh\n')
                writer.writelines('python pretrainer.py \\\n')
                writer.writelines(f'        --model_type {i} \\\n')
                writer.writelines(f'        --model_name_or_path {i} \\\n')
                writer.writelines('        --do_train \\\n')
                writer.writelines('        --do_eval \\\n')
                writer.writelines('        --max_seq_length 128 \\\n')
                writer.writelines('        --per_gpu_train_batch_size 1 \\\n')
                writer.writelines('        --learning_rate 5e-5 \\\n')
                writer.writelines('        --num_train_epochs 10.0 \\\n')
                writer.writelines('        --output_dir result/ \\\n')
                writer.writelines('        --eval_dir evaluation/ \\\n')
                writer.writelines('        --overwrite_output_dir \\\n')
                writer.writelines('        --fp16 \\\n')
                writer.writelines('        --fp16_opt_level O2 \\\n')
                writer.writelines('        --gradient_accumulation_steps 1 \\\n')
                writer.writelines('        --seed 42 \\\n')
                writer.writelines('        --do_lower_case \\\n')
                writer.writelines('        --warmup_steps 100 \\\n')
                writer.writelines('        --logging_steps 100 \\\n')
                writer.writelines('        --save_steps 100 \\\n')
                writer.writelines('        --evaluate_during_training \\\n')
                writer.writelines('        --adam_epsilon 1e-8 \\\n')
                writer.writelines('        --weight_decay 0.05 \\\n')
                writer.writelines('        --max_grad_norm 1.0 \\\n')
                writer.writelines('        --return_token_type_ids \\\n')
                #--for variational use cases
                writer.writelines('        --use_variational_loss \\\n')
                writer.writelines(f'        --vae_model_name {vl} \\\n')
                writer.writelines('        --beta 5.0 \\\n')
                writer.writelines('        --gamma 500.0 \\\n')
                writer.writelines('        --init_kld 1 \\\n')
                writer.writelines('        --init_bce 0.01 \\\n')
                writer.writelines('        --init_mmd 0.01 \\\n')
                writer.writelines('        --latent_dim 100 \\\n')
                if j in exceptions:
                    writer.writelines('        --encoder_decoder \\\n')
                writer.writelines('        --max_steps -1 \n')
                logger.info(f'--job {j}.job is done and recorded')
                writer.writelines('echo -----------------------------------------------\n')
            writer.close()
            logger.info(f"sbatch {job_file}")
            os.system(f"sbatch {job_file}")
        else:
            for mmt in mmd_types:
                job_file = os.path.join(job_directory, f"{i.replace('/', '_').replace('-', '_')}_{vl[:1]}_{mmt[:2]}.job")
                with open(job_file, 'w+') as writer:
                    writer.writelines('#!/bin/bash -l\n')
                    writer.writelines(f'#SBATCH --job-name={j[::-1]}_{vl[:1]}_{mmt[:2]}\n') #reverse model name for anonimity
                    writer.writelines('#SBATCH --mail-user=ifeanyi.ezukwoke@emse.fr\n')
                    writer.writelines('#SBATCH --mail-type=ALL\n')
                    writer.writelines('#SBATCH --gres=gpu:1\n')
                    writer.writelines('#SBATCH --nodes=1\n')
                    writer.writelines('#SBATCH --ntasks=1\n')
                    writer.writelines('#SBATCH --cpus-per-task=4\n')
                    writer.writelines('#SBATCH --mem=50G\n')
                    writer.writelines('#SBATCH --time=7-00:00:00\n')
                    writer.writelines('#SBATCH --partition=audace2018\n')
                    writer.writelines('ulimit -l unlimited\n')
                    writer.writelines('unset SLURM_GTIDS\n')
                    writer.writelines('echo -----------------------------------------------\n')
                    writer.writelines('echo SLURM_NNODES: $SLURM_NNODES\n')
                    writer.writelines('echo SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST\n')
                    writer.writelines('echo SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR\n')
                    writer.writelines('echo SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST\n')
                    writer.writelines('echo SLURM_JOB_ID: $SLURM_JOB_ID\n')
                    writer.writelines('echo SLURM_JOB_NAME: $SLURM_JOB_NAME\n')
                    writer.writelines('echo SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION\n')
                    writer.writelines('echo SLURM_NTASKS: $SLURM_NTASKS\n')
                    writer.writelines('echo SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE\n')
                    writer.writelines('echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE\n')
                    writer.writelines('echo -----------------------------------------------\n')
                    writer.writelines('echo Run program...\n')
                    writer.writelines('source ~/meso-env/env.sh\n')
                    writer.writelines('python pretrainer.py \\\n')
                    writer.writelines(f'        --model_type {i} \\\n')
                    writer.writelines(f'        --model_name_or_path {i} \\\n')
                    writer.writelines('        --do_train \\\n')
                    writer.writelines('        --do_eval \\\n')
                    writer.writelines('        --max_seq_length 128 \\\n')
                    writer.writelines('        --per_gpu_train_batch_size 1 \\\n')
                    writer.writelines('        --learning_rate 5e-5 \\\n')
                    writer.writelines('        --num_train_epochs 10.0 \\\n')
                    writer.writelines('        --output_dir result/ \\\n')
                    writer.writelines('        --eval_dir evaluation/ \\\n')
                    writer.writelines('        --overwrite_output_dir \\\n')
                    writer.writelines('        --fp16 \\\n')
                    writer.writelines('        --fp16_opt_level O2 \\\n')
                    writer.writelines('        --gradient_accumulation_steps 1 \\\n')
                    writer.writelines('        --seed 42 \\\n')
                    writer.writelines('        --do_lower_case \\\n')
                    writer.writelines('        --warmup_steps 100 \\\n')
                    writer.writelines('        --logging_steps 100 \\\n')
                    writer.writelines('        --save_steps 100 \\\n')
                    writer.writelines('        --evaluate_during_training \\\n')
                    writer.writelines('        --adam_epsilon 1e-8 \\\n')
                    writer.writelines('        --weight_decay 0.05 \\\n')
                    writer.writelines('        --max_grad_norm 1.0 \\\n')
                    writer.writelines('        --return_token_type_ids \\\n')
                    #--for variational use cases
                    writer.writelines('        --use_variational_loss \\\n')
                    writer.writelines(f'        --vae_model_name {vl} \\\n')
                    writer.writelines(f'        --mmd_type {mmt} \\\n')
                    writer.writelines('        --beta 5.0 \\\n')
                    writer.writelines('        --gamma 500.0 \\\n')
                    writer.writelines('        --init_kld 1 \\\n')
                    writer.writelines('        --init_bce 0.01 \\\n')
                    writer.writelines('        --init_mmd 0.01 \\\n')
                    writer.writelines('        --latent_dim 100 \\\n')
                    if j in exceptions:
                        writer.writelines('        --encoder_decoder \\\n')
                    writer.writelines('        --max_steps -1 \n')
                    logger.info(f'--job {j}.job is done and recorded')
                    writer.writelines('echo -----------------------------------------------\n')
                writer.close()
                logger.info(f"sbatch {job_file}")
                os.system(f"sbatch {job_file}")