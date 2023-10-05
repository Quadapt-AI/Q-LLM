import os
import glob
import numpy as np
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
path = os.getcwd()
year = 2019
absoulte_dir = [join(path, f'plm/finetuning/{year}'), join(path, f'plm/use_weight/{year}')]
MODEL_CLASSES = {
                    'facebook/bart-large-cnn': 'bart',
                    'bert-base-uncased': 'bert', 
                    'roberta-base': 'roberta', 
                    # 'EleutherAI': 'gpt3',
                    'gpt2': 'gpt2-b',
                    'gpt2-medium': 'gpt2-m',
                    'gpt2-large': 'gpt2-l',
                    }

result_files = {}
for dir_ in absoulte_dir:
    for i, j in MODEL_CLASSES.items():
        if dir_.split('/')[-2].lower() == 'finetuning':
            result_files[f'{j}'] = join(dir_, f"{i.split('/')[0]}/evaluation")
        elif dir_.split('/')[-2].lower() == 'use_weight':
            result_files[f'w_{j}'] = join(dir_, f"{i.split('/')[0]}/evaluation")
        else:
            result_files[f"{dir_.split('/')[-2].lower()[:3]}_{j}"] = join(dir_, f"{i.split('/')[0]}/evaluation")

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

    
    
#%% Plotting lossses and perplexity



keys_ = [x for x in list(eval_metric.keys()) if not 'w_' in x] # list of model name
wkeys_ = [x for x in list(eval_metric.keys()) if 'w_' in x] # list of model name
col = ['red', 'green', 'blue', 'brown', 'purple', 'pink', 'orange']
ind = {
       'training_loss': 'Training loss',
       'evaluation_loss': 'Evaluation loss',
       'perplexity': 'Perplexity',
       }

char = range(97, 97+len(ind.keys())+1)
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
axes = axes.ravel()

char = range(97, 97+len(axes)+1)

for (_, ax), (k, v), z in zip(enumerate(axes), ind.items(), char):
    for i, j, w in zip(keys_, wkeys_, col):
        x_ = eval_metric[i] # using the model to extract their dictionary values
        
        ind_ = ''.join([x for x in list(x_.keys()) if k in x])
        wx_ = eval_metric[j] # using the model to extract their dictionary values
        ax.plot(range(1, len(x_[ind_])+1), x_[ind_], label = f'{i.upper()}', lw = 1.7, marker='s', c = w) # plot the loss or eval metric
        ax.plot(range(1, len(wx_[ind_])+1), wx_[ind_], label = f'w-{i.upper()}', lw = 1.7, linestyle = '--', marker='o', c = w) # plot the loss or eval metric
        ax.set_xlabel(f'Number of steps\n\n({chr(z)}) {v}', fontsize=20)
        ax.set_ylabel(f'{v}', fontsize=20)
        ax.legend()
        ax.grid(linewidth=0.2)
        ax.legend(loc='upper center', bbox_to_anchor = (0.5, 1.19), ncol = 4, fancybox = False, shadow = False, fontsize = 10)

plt.tight_layout()

#%%
from matplotlib.offsetbox import AnchoredText

keys_ = [x for x in list(eval_metric.keys()) if not 'w_' in x] # list of model name
wkeys_ = [x for x in list(eval_metric.keys()) if 'w_' in x] # list of model name

col = ['red', 'green', 'blue', 'brown', 'purple', 'pink', 'orange']
ind = {
        'bleuscore_': 'BLEU-1',
        'bleuscore3_': 'BLEU-3',
        'lese1_': 'LESE-1',
        'lese3_': 'LESE-3',
       }



leny_ = len(list(MODEL_CLASSES.values())) #number of models to plot
lenx_ = len(list(ind.values())) #number of metrics to plot
fig, ax = plt.subplots(leny_, lenx_, figsize=(8, 3))

char = range(97, 97+len(ind.keys())+1)
for (en_, (k, v)), z in zip(enumerate(ind.items()), char):
    for (en, _), (i, j) in zip(enumerate(ax), zip(keys_, wkeys_)):
        x_ = eval_metric[i] # using the model to extract their dictionary values
        ind_ = ''.join([x for x in list(x_.keys()) if k in x])
        if ind_ != '':
            wx_ = eval_metric[j] # using the model to extract their dictionary values
            if k in list(ind.keys())[:2]:
                ax[en, en_].hist(x_[ind_], bins = 80, label = f'{i.upper()}', alpha = 0.7, color = 'r')
                ax[en, en_].hist(wx_[ind_], bins = 80, ls='dotted', lw = 1.7, label = f'w-{i.upper()}', alpha = 0.4, color = 'b')
                anchored_text = AnchoredText(rf'$\mu:{round(np.mean(x_[ind_]), 2)},\ \sigma:{round(np.std(x_[ind_]), 2)}$'+'\n'+\
                                             rf'$\mu_w:{round(np.mean(wx_[ind_]), 2)},\ \sigma_w:{round(np.std(wx_[ind_]), 2)}$', loc=4, prop={'fontsize':5},
                                             frameon=False, )
                ax[en, en_].add_artist(anchored_text, )
            else:
                if k in list(ind.keys())[2:4]:
                    ax[en, en_].hist(x_[ind_]['fs_lev'], bins = 80, label = f'{i.upper()}', alpha = 0.7, color = 'r')
                    ax[en, en_].hist(wx_[ind_]['fs_lev'], bins = 80, ls='dotted', lw = 1.7, label = f'w-{i.upper()}', alpha = 0.4, color = 'b')
                    anchored_text = AnchoredText(rf"$\mu={round(np.mean(x_[ind_]['fs_lev']), 2)},\ \sigma={round(np.std(x_[ind_]['fs_lev']), 2)}$"+'\n'+\
                                             rf"$\mu_w={round(np.mean(wx_[ind_]['fs_lev']), 2)},\ \sigma_w={round(np.std(wx_[ind_]['fs_lev']), 2)}$", loc=4, prop={'fontsize':5},
                                             frameon=False, )
                    ax[en, en_].add_artist(anchored_text)
                else:
                    if v == 'ROUGE-1':
                        ax[en, en_].hist(x_[ind_]['rouge-1']['f'], bins = 80, label = f'{i.upper()}', alpha = 0.7, color = 'r')
                        ax[en, en_].hist(wx_[ind_]['rouge-1']['f'], bins = 80, ls='dotted', lw = 1.7, label = f'w-{i.upper()}', alpha = 0.4, color = 'b')
                    elif v == 'ROUGE-L':
                        ax[en, en_].hist(x_[ind_]['rouge-l']['f'], bins = 80, label = f'{i.upper()}', alpha = 0.7, color = 'r')
                        ax[en, en_].hist(wx_[ind_]['rouge-l']['f'], bins = 80, ls='dotted', lw = 1.7, label = f'w-{i.upper()}', alpha = 0.4, color = 'b')
        else:
            ax[en, en_].hist(np.array([0]*50), bins = 50, label = f'{i.upper()}', alpha = 0.7, color = 'r')
            ax[en, en_].hist(np.array([0]*50), bins = 80, ls='dotted', lw = 1.7, label = f'w-{i.upper()}', alpha = 0.4, color = 'b')
        ax[len(keys_)-1, en_].set_xlabel(f'Number of steps\n\n({chr(z)}) {v}')
        ax[en, 0].set_ylabel(f'{i.upper()}', fontsize=10)
        ax[en, en_].legend()
        ax[en, en_].grid(linewidth=0.2)
        ax[en, en_].legend(fancybox = False, shadow = False, fontsize = 8)


















