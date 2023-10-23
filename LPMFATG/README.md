# Leveraging-Pre-trained-Models-for-Failure-Analysis-Triplets-Generation

### ABSTRACT
```
Pre-trained Language Models recently gained traction in the Natural Language Processing (NLP)
domain for text summarization, generation and question answering tasks. This stems from the
innovation introduced in Transformer models and their overwhelming performance compared with
Recurrent Neural Network Models (Long Short Term Memory (LSTM)). In this paper, we leverage
the attention mechanism of pre-trained causal language models such as Transformer model for
the downstream task of generating Failure Analysis Triplets (FATs) - a sequence of steps for
analyzing defected components in the semiconductor industry. We compare different transformer
model for this generative task and observe that Generative Pre-trained Transformer 2 (GPT2)
outperformed other transformer model for the failure analysis triplet generation (FATG) task.
In particular, we observe that GPT2 (trained on 1.5B parameters) outperforms pre-trained 
BERT, BART and GPT3 by a large margin on ROUGE. Furthermore, we introduce LEvenshstein
Sequential Evaluation metric (LESE) for better evaluation of the structured FAT data and
show that it compares exactly with human judgment than existing metrics.
```

------------------------------

### How to use

 - Clone the repository: ```git clone https://github.com/AI-for-Fault-Analysis-FA4-0/Leveraging-Pre-trained-Models-for-Failure-Analysis-Triplets-Generation```
 - Run training and evaluation example
    - ```python ```\
            ```python pretrainer.py  ```\
            ```--model_type gpt2 ``` \
            ```--model_name_or_path gpt2 ``` \
            ```--do_train ``` \
            ```--do_eval ``` \
            ```--max_seq_length ```  ```128 ``` \
            ```--per_gpu_train_batch_size ```  ```1 ``` \
            ```--learning_rate ```  ```5e-5 ``` \
            ```--num_train_epochs ```  ```5.0 ``` \
            ```--output_dir ```  ```result/ ``` \
            ```--eval_dir ```  ```evaluation/ ``` \
            ```--overwrite_output_dir ``` \
            ```--fp16 ``` \
           ``` --fp16_opt_level ```  ```O2 ``` \
            ```--gradient_accumulation_steps ```  ```1 ``` \
            ```--seed ```  ```42 ``` \
            ```--do_lower_case ``` \
            ```--warmup_steps ```  ```100 ``` \
            ```--logging_steps ```  ```100 ``` \
            ```--save_steps ```  ```100 ``` \
           ``` --evaluate_during_training ``` \
            ```--save_total_limit ```  ```1 ``` \
            ```--adam_epsilon ```  ```1e-8 ``` \
            ```--weight_decay ```  ```0.05 ``` \
            ```--max_grad_norm ```  ```1.0 ``` \
            ```--return_token_type_ids ``` \
            ```#--use_weights ``` \
            ```--max_steps ```  ```-1 ```
           ```
- Model type/name with Causal LMHead
  - ```facebook/bart-large-cnn```: Bidirectional Auto-Regressive Transformer
  - ```bert-base-uncased```: Bidirectional Encoder Representations from Transformers
  - ```roberta-large```: Robustly Optimized BERT Pretraining Approach
  - ```distilbert-base-uncased```: A distilled version of BERT: smaller, faster, cheaper and lighter
  - ```xlnet-large-cased```: Generalized Autoregressive Pretraining for Language Understanding
  - ```openai-gpt```: Generative Pre-trained Transformer 3
  - ```gpt2```: Generative Pre-trained Transformer 2 (base)
  - ```gpt2-medium```: Generative Pre-trained Transformer 2 (Medium)
  - ```gpt2-large```: Generative Pre-trained Transformer 2 (Large)

------------------------------

### Results

#### Cross-entropy loss & co.

| Model     | BLEU-1 | BLEU-3 | MET.  | ROUGE-1 |  |  |ROUGE-L  |  |  | LESE-1   |       |       |   Lev-1    |   LESE-3    |       |      |   Lev-3   |        PPL            |
|-----------|--------|--------|-------|---------|---------|--------|-------|--------|-------|-------|-------|-------|-------|-------|-------|------|------|--------------------|
|           |        |        |       | Prec.   | Rec.    | F1     | Prec. | Rec.   | F1    | Prec. | Rec.  | F1    |       | Prec. | Rec.  | F1   |      |                    |
| BART      | 3.04   | 1.67   | 3.23  | -       | -       | -      | -     | -      | -     | 2.12  | 3.85  | 2.28  | 73.0  | 0.01  | 0.0   | 0.0  | 24.0 | 1.0                |
| BERT      | 1.81   | 0.65   | 4.0   | 6.7     | 12.0    | 7.99   | 5.62  | 10.21  | 6.72  | 1.38  | 10.48 | 2.33  | 287.0 | 0.02  | 0.03  | 0.01 | 96.0 | 1.0                |
| ROBERTA   | 0.14   | 0.11   | 0.32  | 0.26    | 0.56    | 0.34   | 0.26  | 0.55   | 0.34  | 0.09  | 0.33  | 0.13  | 169.0 | 0.0   | 0.0   | 0.0  | 56.0 | 1.0                |
| GPT3      | 22.71  | 16.6   | 29.34 | 30.06   | 35.75   | 30.26  | 27.65 | 32.93  | 27.83 | 20.88 | 24.93 | 20.64 | 45.0  | 9.34  | 11.28 | 9.29 | 16.0 | 1.53 |
| GPT2-B    | 20.85  | 15.25  | 25.67 | 30.66   | 31.86   | 28.78  | 28.1  | 29.2   | 26.35 | 21.31 | 21.1  | 19.17 | 41.0  | 9.31  | 9.3   | 8.39 | 15.0 | 1.42 |
| GPT2-M    | 21.26  | 15.47  | 26.74 | 30.37   | 33.28   | 29.15  | 27.65 | 30.4   | 26.56 | 21.08 | 22.06 | 19.41 | 43.0  | 9.23  | 9.79  | 8.55 | 15.0 | 1.52 |
| GPT2-L    | 22.87  | 16.87  | 28.7  | 31.88   | 35.19   | 30.87  | 29.19 | 32.24  | 28.24 | 22.01 | 23.83 | 20.81 | 42.0  | 10.06 | 10.89 | 9.53 | 15.0 | 1.412 |
| W_BART    | 4.71   | 1.87   | 5.04  | 5.41    | 11.05   | 6.57   | 4.36  | 8.97   | 5.28  | 2.56  | 5.9   | 3.17  | 81.0  | 0.0   | 0.0   | 0.0  | 27.0 | 1.01 |
| W_BERT    | 0.42   | 0.2    | 0.56  | 1.08    | 1.49    | 1.15   | 0.97  | 1.37   | 1.04  | 0.33  | 1.15  | 0.44  | 74.0  | 0.01  | 0.0   | 0.0  | 24.0 | 1.0                |
| W_ROBERTA | 0.07   | 0.06   | 0.33  | 0.11    | 0.32    | 0.16   | 0.11  | 0.31   | 0.15  | 0.05  | 0.2   | 0.07  | 196.0 | 0.0   | 0.0   | 0.0  | 65.0 | 1.0                |
| W_GPT3    | -      | -      | -     | -       | -       | -      | -     | -      | -     | -     | -     | -     | -     | -     | -     | -    | -    | 1.34  |
| W_GPT2-B  | 21.15  | 15.52  | 26.21 | 31.27   | 32.49   | 29.33  | 28.51 | 29.64  | 26.72 | 21.64 | 21.52 | 19.49 | 41.0  | 9.59  | 9.62  | 8.65 | 15.0 | 1.28 |
| W_GPT2-M  | 20.99  | 15.38  | 26.34 | 30.05   | 32.53   | 28.68  | 27.44 | 29.76  | 26.2  | 21.0  | 21.68 | 19.28 | 42.0  | 9.43  | 9.75  | 8.67 | 15.0 | 1.34  |
| W_GPT2-L  | 22.67  | 16.64  | 28.68 | 31.66   | 35.54   | 30.89  | 29.02 | 32.54  | 28.26 | 21.93 | 24.09 | 20.8  | 43.0  | 10.1  | 11.17 | 9.62 | 16.0 | 1.26 |

#### Variational Loss

| Model                  | BLEU-1 | BLEU-3 | MET.  | ROUGE-1 |  |  |  ROUGE-L|  |  | LESE-1  |       |       |     Lev-1  |    LESE-3   |      |      |   Lev-3   |          PPL          |
|------------------------|--------|--------|-------|---------|---------|--------|-------|--------|-------|-------|-------|-------|-------|-------|------|------|------|--------------------|
|                        |        |        |       | Prec.   | Rec.    | F1     | Prec. | Rec.   | F1    | Prec. | Rec.  | F1    |       | Prec. | Rec. | F1   |      |                    |
| BETAVAE_MAH_BART       | 1.96   | 1.37   | 1.7   | 4.86    | 3.4     | 3.63   | 4.56  | 3.19   | 3.39  | 1.41  | 1.95  | 1.47  | 54.0  | 0.0   | 0.0  | 0.0  | 18.0 | 1.44  |
| BETAVAE_MAH_BERT       | 9.91   | 3.56   | 8.78  | 27.17   | 21.61   | 20.77  | 23.24 | 18.49  | 17.7  | 10.4  | 9.01  | 7.53  | 49.0  | 0.94  | 0.37 | 0.36 | 17.0 | 1.99 |
| BETAVAE_MAH_ROBERTA    | 0.34   | 0.2    | 0.53  | 0.66    | 1.35    | 0.81   | 0.61  | 1.24   | 0.75  | 0.24  | 1.03  | 0.35  | 265.0 | 0.0   | 0.0  | 0.0  | 88.0 | 1.94  |
| BETAVAE_MAH_GPT2-B     | 15.18  | 5.53   | 12.17 | 21.7    | 22.05   | 20.09  | 19.28 | 19.66  | 17.85 | 11.98 | 12.17 | 10.83 | 45.0  | 0.29  | 0.33 | 0.27 | 16.0 | 1.97 |
| BETAVAE_MAH_GPT2-M     | 14.52  | 5.14   | 12.19 | 18.74   | 22.03   | 18.52  | 16.52 | 19.48  | 16.32 | 10.26 | 12.71 | 10.18 | 50.0  | 0.3   | 0.41 | 0.31 | 18.0 | 2.05  |
| BETAVAE_MAH_GPT2-L     | 13.95  | 5.0    | 11.68 | 18.55   | 21.8    | 18.23  | 16.32 | 19.24  | 16.05 | 10.56 | 12.27 | 10.06 | 50.0  | 0.27  | 0.38 | 0.28 | 18.0 | 1.94  |
| GCVAE_MMD_BART         | 4.66   | 1.8    | 4.31  | 6.76    | 12.74   | 7.5    | 5.03  | 9.11   | 5.35  | 2.94  | 4.3   | 2.82  | 57.0  | 0.16  | 0.01 | 0.03 | 19.0 | 1.78 |
| GCVAE_MMD_BERT         | 9.54   | 3.32   | 8.51  | 21.61   | 20.21   | 18.31  | 18.29 | 17.25  | 15.5  | 8.16  | 9.14  | 6.87  | 54.0  | 0.42  | 0.18 | 0.16 | 19.0 | 2.46 |
| GCVAE_MMD_ROBERTA      | 1.85   | 0.86   | 2.43  | 3.82    | 6.44    | 4.17   | 3.29  | 5.61   | 3.59  | 1.58  | 4.28  | 1.84  | 143.0 | 0.07  | 0.01 | 0.01 | 48.0 | 2.45  |
| GCVAE_MMD_GPT2-B       | 15.12  | 5.21   | 11.65 | 21.42   | 21.43   | 19.69  | 18.88 | 18.99  | 17.38 | 11.61 | 11.83 | 10.56 | 44.0  | 0.23  | 0.26 | 0.22 | 16.0 | 2.53 |
| GCVAE_MMD_GPT2-M       | 14.44  | 4.79   | 12.09 | 18.66   | 22.74   | 18.7   | 16.4  | 20.08  | 16.44 | 10.02 | 13.5  | 10.21 | 54.0  | 0.18  | 0.29 | 0.2  | 19.0 | 2.59 |
| GCVAE_MMD_GPT2-L       | 13.95  | 4.45   | 12.56 | 17.5    | 24.57   | 18.52  | 15.17 | 21.37  | 16.06 | 9.37  | 14.86 | 10.12 | 62.0  | 0.16  | 0.23 | 0.17 | 22.0 | 2.40 |
| GCVAE_MAH_BART         | 4.22   | 1.79   | 3.58  | 6.91    | 10.26   | 6.96   | 5.24  | 7.45   | 5.08  | 2.82  | 3.41  | 2.53  | 50.0  | 0.09  | 0.01 | 0.02 | 17.0 | 1.78 |
| GCVAE_MAH_BERT         | 9.35   | 3.38   | 7.5   | 23.8    | 17.39   | 17.99  | 19.74 | 14.73  | 15.04 | 8.32  | 7.94  | 6.69  | 47.0  | 0.45  | 0.13 | 0.14 | 16.0 | 2.46 |
| GCVAE_MAH_ROBERTA      | 1.18   | 0.64   | 1.7   | 1.18    | 3.68    | 1.72   | 1.06  | 3.33   | 1.54  | 0.78  | 2.85  | 1.16  | 142.0 | 0.01  | 0.04 | 0.01 | 47.0 | 2.46 |
| GCVAE_MAH_GPT2-B       | 15.7   | 5.36   | 12.18 | 21.82   | 22.24   | 20.29  | 19.23 | 19.72  | 17.91 | 11.68 | 12.34 | 10.81 | 45.0  | 0.25  | 0.28 | 0.23 | 16.0 | 2.53 |
| GCVAE_MAH_GPT2-M       | 14.34  | 4.81   | 11.89 | 19.06   | 22.28   | 18.75  | 16.7  | 19.66  | 16.46 | 10.32 | 13.1  | 10.2  | 52.0  | 0.19  | 0.27 | 0.2  | 19.0 | 2.58 |
| GCVAE_MAH_GPT2-L       | 14.05  | 4.75   | 11.53 | 18.59   | 21.96   | 18.38  | 16.19 | 19.23  | 16.03 | 10.13 | 12.44 | 9.94  | 51.0  | 0.2   | 0.26 | 0.2  | 18.0 | 2.39  |
| CONTROLVAE_MAH_BART    | 4.66   | 1.98   | 3.6   | 7.42    | 9.69    | 7.42   | 5.48  | 6.94   | 5.29  | 2.57  | 2.57  | 2.09  | 47.0  | 0.21  | 0.02 | 0.04 | 16.0 | 1.78 |
| CONTROLVAE_MAH_BERT    | 7.91   | 2.91   | 7.19  | 18.97   | 17.7    | 15.82  | 15.88 | 15.11  | 13.3  | 7.13  | 8.4   | 5.97  | 57.0  | 0.32  | 0.17 | 0.13 | 20.0 | 2.46 |
| CONTROLVAE_MAH_ROBERTA | 1.17   | 0.52   | 1.96  | 1.38    | 4.7     | 2.04   | 1.2   | 4.17   | 1.78  | 0.87  | 4.24  | 1.38  | 200.0 | 0.0   | 0.01 | 0.0  | 67.0 | 2.45  |
| CONTROLVAE_MAH_GPT2-B  | 15.83  | 5.34   | 12.43 | 21.97   | 22.66   | 20.51  | 19.34 | 20.06  | 18.09 | 11.6  | 12.75 | 10.92 | 46.0  | 0.23  | 0.3  | 0.23 | 17.0 | 2.54 |
| CONTROLVAE_MAH_GPT2-M  | 14.56  | 4.78   | 12.24 | 18.94   | 23.02   | 18.95  | 16.61 | 20.29  | 16.64 | 10.34 | 13.68 | 10.43 | 53.0  | 0.17  | 0.26 | 0.19 | 19.0 | 2.58 |
| CONTROLVAE_MAH_GPT2-L  | 13.77  | 4.72   | 11.22 | 18.43   | 21.1    | 17.91  | 16.09 | 18.46  | 15.64 | 10.16 | 11.86 | 9.67  | 50.0  | 0.19  | 0.21 | 0.17 | 18.0 | 2.39  |
| VAE_MAH_BART           | 2.8    | 1.5    | 3.89  | 29.29   | 9.75    | 13.63  | 22.94 | 7.76   | 10.71 | 19.24 | 4.41  | 6.77  | 38.0  | 3.11  | 0.35 | 0.61 | 13.0 | 4.87  |
| VAE_MAH_BERT           | 0.15   | 0.08   | 0.4   | 34.72   | 5.42    | 8.48   | 30.04 | 4.32   | 6.82  | 5.31  | 0.51  | 0.87  | 39.0  | inf   | 0.18 | nan  | 13.0 | 6.68  |
| VAE_MAH_ROBERTA        | 2.34   | 1.0    | 2.67  | 7.64    | 7.51    | 5.94   | 6.94  | 6.98   | 5.44  | 3.36  | 5.19  | 2.7   | 79.0  | 0.38  | 0.06 | 0.08 | 27.0 | 7.0                |
| VAE_MAH_GPT2-B         | 16.27  | 5.52   | 11.65 | 33.56   | 19.81   | 23.16  | 27.29 | 16.48  | 19.06 | 15.56 | 10.45 | 11.47 | 38.0  | 0.17  | 0.11 | 0.12 | 14.0 | 9.01  |
| VAE_MAH_GPT2-M         | 14.33  | 4.74   | 11.0  | 29.92   | 19.76   | 21.23  | 24.71 | 16.69  | 17.75 | 15.92 | 11.21 | 11.37 | 42.0  | 0.25  | 0.12 | 0.12 | 15.0 | 7.58  |
| VAE_MAH_GPT2-L         | 14.25  | 4.6    | 10.74 | 21.72   | 19.39   | 18.69  | 18.19 | 16.45  | 15.73 | 10.56 | 11.3  | 9.68  | 47.0  | 0.09  | 0.11 | 0.08 | 17.0 | 6.02  |


### Cite

```
@misc{https://doi.org/10.48550/arxiv.2210.17497,
  doi = {10.48550/ARXIV.2210.17497},
  url = {https://arxiv.org/abs/2210.17497},
  author = {Ezukwoke, Kenneth and Hoayek, Anis and Batton-Hubert, Mireille and Boucher, Xavier and Gounet, Pascal and Adrian, Jerome},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), Applications (stat.AP), FOS: Computer and information sciences, FOS: Computer and information sciences, G.3; I.2; I.7, 68Txx, 68Uxx},
  title = {Leveraging Pre-trained Models for Failure Analysis Triplets Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
