import os
import glob
import numpy as np
from os.path import join
import numpy as np
path = os.getcwd()
year = 2019
absoulte_dir = join(path, f'plm/vfinetuning/')
MODEL_CLASSES = {
                    'facebook': 'bart',
                    'bert-base-uncased': 'bert', 
                    'roberta-base': 'roberta', 
                    # 'EleutherAI': 'gpt3',
                    'gpt2': 'gpt2-b',
                    'gpt2-medium': 'gpt2-m',
                    'gpt2-large': 'gpt2-l',
                    }

vae_loss_typ = ['betavae', 'gcvae', 'controlvae', 'vae']

result_files = {}
for vl in vae_loss_typ:
    kernels_ = os.listdir(join(absoulte_dir, f'{vl}'))
    # print(kernels_)
    for kl in kernels_:
        for i, j in MODEL_CLASSES.items():
            ev_dir_ = join(join(absoulte_dir, f'{vl}/{kl}'), f'{year}/{i}/evaluation')
            # print(ev_dr_)
            ev_dr_= ev_dir_.split('/')
            result_files[f"{vl}_{ev_dr_[-4]}_{j}"] = ev_dir_



eval_metric = {}
for i, j in result_files.items():
    npys = [x for x in os.listdir(j) if '.npy' in x]
    eval_metric[i] = {}
    for npy in npys:
        if not 'lese' in npy:
            eval_metric[i][f"{npy.split('.npy')[0]}"] = np.load(join(j, npy), allow_pickle = True)
        else:
            eval_metric[i][f"{npy.split('.npy')[0]}"] = np.load(join(j, npy), allow_pickle = True).ravel()[0]
        if 'rouge' in npy:
            eval_metric[i][f"{npy.split('.npy')[0]}"] = np.load(join(j, npy), allow_pickle = True).ravel()[0]

print('+-------------+-------------+-------------+-------------+--------------------+--------------------+--------------------+-------------+--------------------+---------------+---------------+')
print('|  Model      |   BLEU-1    |    BLEU-3   |   MET.      |      ROUGE-1       |      ROUGE-L       |        LESE-1      |    Lev-1    |  LESE-3            |   Lev-3       |      PPL      |')
print('|            |              |             |             | Prec. | Rec. | F1  | Prec. | Rec. | F1  | Prec. | Rec. | F1  |             | Prec. | Rec. | F1  |               |               |')
print('+------------+-------------+-------------+-------------+-------+------+-----+--------+------+-----+-------+------+-----+-------------+-------+------+-----+---------------+---------------+')
for i, j in eval_metric.items():
    ppl = round(j['perplexity'][-1], 2)
    bleuscore_ = ''.join([x for x in j.keys() if 'bleuscore_' in x]) if bool([x for x in j.keys() if 'bleuscore_' in x]) else '-'
    bleuscore_ = round(np.mean(j[bleuscore_])*100, 2) if bleuscore_ != '-' else '-'
    bleuscore3_ = ''.join([x for x in j.keys() if 'bleuscore3_' in x]) if bool([x for x in j.keys() if 'bleuscore3_' in x]) else '-'
    bleuscore3_ = round(np.mean(j[bleuscore3_])*100, 2) if bleuscore3_ != '-' else '-'
    meteor_ = ''.join([x for x in j.keys() if 'meteor_' in x]) if bool([x for x in j.keys() if 'meteor_' in x]) else '-'
    meteor_ = round(np.mean(j[meteor_])*100, 2) if meteor_ != '-' else '-'
    rouge_ = ''.join([x for x in j.keys() if 'rouge_' in x]) if bool([x for x in j.keys() if 'rouge_' in x]) else '-'
    rouge_ = j[rouge_] if rouge_ != '-' else '-'
    rouge1_ = rouge_['rouge-1'] if rouge_ != '-' else '-'
    p1_, r1_, f1_1_ = round(rouge1_['p']*100, 2) if rouge1_ != '-' else '-', round(rouge1_['r']*100, 2) if rouge1_ != '-' else '-', \
                        round(rouge1_['f']*100, 2) if rouge1_ != '-' else '-'
    rougel_ = rouge_['rouge-l'] if rouge_ != '-' else '-'
    pl_, rl_, f1_l_ = round(rougel_['p']*100, 2) if rougel_ != '-' else '-', round(rougel_['r']*100, 2) if rougel_ != '-' else '-', \
                        round(rougel_['f']*100, 2) if rougel_ != '-' else '-'
    lese_ = ''.join([x for x in j.keys() if 'lese1_' in x]) if bool([x for x in j.keys() if 'lese1_' in x]) else '-'
    lese_ = j[lese_] if lese_ != '-' else '-'
    pls_, rls_, f1_ls_, levd1_ = round(np.mean(lese_['prec_lev'])*100, 2) if lese_ != '-' else '-', \
                                    round(np.mean(lese_['rec_lev'])*100, 2) if lese_ != '-' else '-', \
                                    round(np.mean(lese_['fs_lev'])*100, 2) if lese_ != '-' else '-', \
                                    np.mean(lese_['lev_d'])//1 if lese_ != '-' else '-'
    lese3_ = ''.join([x for x in j.keys() if 'lese3_' in x]) if bool([x for x in j.keys() if 'lese3_' in x]) else '-'
    lese3_ = j[lese3_] if lese3_ != '-' else '-'
    pls3_, rls3_, f1_ls3_, levd3_ = round(np.mean(lese3_['prec_lev'])*100, 2) if lese3_ != '-' else '-', \
                                    round(np.mean(lese3_['rec_lev'])*100, 2) if lese3_ != '-' else '-', \
                                    round(np.mean(lese3_['fs_lev'])*100, 2) if lese3_ != '-' else '-', \
                                    np.mean(lese3_['lev_d'])//3 if lese3_ != '-' else '-'
    print(f"|  {i.upper()}    |   {bleuscore_}   |   {bleuscore3_}  | {meteor_}  |  {p1_} |  {r1_}  |  {f1_1_}  |  {pl_} |  {rl_} |  {f1_l_} |  {pls_}  | {rls_}  |  {f1_ls_}  |  {levd1_}  |  {pls3_}  |  {rls3_}  |  {f1_ls3_}  |  {levd3_}  |   {ppl}  |")
print('+------------+-------------+-------------+-------------+-------+------+-----+--------+------+-----+-------+------+-----+-------------+-------+------+-----+---------------+---------------+')        

    
    
#%% Plotting graphs

import itertools
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

mod = list(MODEL_CLASSES.values())
keys_ = [x for x in list(eval_metric.keys()) if not 'w_' in x] # list of model name
dict_k = {
            'betavae_mah_bart': r'$\beta-BART$',
            'betavae_mah_bert': r'$\beta-BERT$',
            'betavae_mah_roberta': r'$\beta-ROBERTA$',
            'betavae_mah_gpt2-b': r'$\beta-GPT2-B$',
            'betavae_mah_gpt2-m': r'$\beta-GPT2-M$',
            'betavae_mah_gpt2-l': r'$\beta-GPT2-L$',
            'gcvae_mmd_bart': r'$GCVAE-BART$',
            'gcvae_mmd_bert': r'$GCVAE-BERT$',
            'gcvae_mmd_roberta': r'$GCVAE-ROBERTA$',
            'gcvae_mmd_gpt2-b': r'$GCVAE-GPT2-B$',
            'gcvae_mmd_gpt2-m': r'$GCVAE-GPT2-M$',
            'gcvae_mmd_gpt2-l': r'$GCVAE-GPT2-L$',
            'gcvae_mah_bart': r'$GCVAE-BART$',
            'gcvae_mah_bert': r'$GCVAE-BERT$',
            'gcvae_mah_roberta': r'$GCVAE-ROBERTA$',
            'gcvae_mah_gpt2-b': r'$GCVAE-GPT2-B$',
            'gcvae_mah_gpt2-m': r'$GCVAE-GPT2-M$',
            'gcvae_mah_gpt2-l': r'$GCVAE-GPT2-L$',
            'controlvae_mah_bart': r'$\beta_t-BART$',
            'controlvae_mah_bert': r'$\beta_t-BERT$',
            'controlvae_mah_roberta': r'$\beta_t-ROBERTA$',
            'controlvae_mah_gpt2-b': r'$\beta_t-GPT2-B$',
            'controlvae_mah_gpt2-m': r'$\beta_t-GPT2-M$',
            'controlvae_mah_gpt2-l': r'$\beta_t-GPT2-L$',
            'vae_mah_bart': r'$VAE-BART$',
            'vae_mah_bert': r'$VAE-BERT$',
            'vae_mah_roberta': r'$VAE-ROBERTA$',
            'vae_mah_gpt2-b': r'$VAE-GPT2-B$',
            'vae_mah_gpt2-m': r'$VAE-GPT2-M$',
            'vae_mah_gpt2-l': r'$VAE-GPT2-L$',
    }

vae_loss_typ = {
                'vae_mah_': 'VAE',
                'betavae_': r'$\beta$-VAE',
                'controlvae_': 'ControlVAE', 
                'gcvae_mmd_': 'GCVAE-MMD',
                'gcvae_mah_': 'GCVAE-MAH', 
                
                
                }
col = ['red', 'green', 'blue', 'brown', 'purple', 'pink']
ind = {
        'training_loss': 'Training loss',
        'evaluation_loss': 'Evaluation loss',
        'perplexity': 'Perplexity',
        'training_kl': 'KL loss',
        'training_alpha': r'$\alpha$',
        'training_beta': r'$\beta$',
        'training_gamma': r'$\gamma$',
       }

fig, ax = plt.subplots(len(vae_loss_typ.keys()), len(ind.values()), figsize=(8, 3))
marker = itertools.cycle(('^', '<', '>', 's', '8', 'p'))


char = range(97, 97+len(ind.keys())+1)
for (en_, (k, v)), z in zip(enumerate(ind.items()), char):
    for (en, _), (i, j) in zip(enumerate(ax), vae_loss_typ.items()):
        if i != 'vae_mah_':
            mod_ = [x for x in dict_k.keys() if i in x]
        else:
            mod_ = [x for x in dict_k.keys() if i in x][-6:]
        for md_, c in zip(mod_, col):
            x_ = eval_metric[md_] # using the model to extract their dictionary values
            ind_ = ''.join([x for x in list(x_.keys()) if k in x])
            ax[en, en_].plot(range(1, len(x_[ind_])+1), x_[ind_], label = dict_k[md_], lw = 0.7, markeredgecolor='none', marker=next(marker), markersize=4)
            ax[en, 0].set_ylabel(j, fontsize=10)
            ax[len(vae_loss_typ.keys())-1, en_].set_xlabel(f'Number of steps\n\n({chr(z)}) {v}')
            ax[en, en_].legend()
            ax[en, en_].grid(linewidth=0.2)
            ax[en, en_].legend(fancybox = False, shadow = False, fontsize = 5)
            
            
            
                


