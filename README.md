# Turkish Treebank Benchmarking

This repo is for benchmarking Transformers on Turkish treebanks, more openly POS-DEP-MORPH tasks. 

There are 2 scripts, `run.sh` that you can give the parameters and real trainer script `train_pos_dep_morph.py`. The second one is based on HF code, Transformer, Tokenizer and Trainer code as well as `datasets` to pull the treebanks from HF. Our [HF repo](https://huggingface.co/datasets/BayanDuygu/Treebank-Benchmarking) includes 2 treebanks as well, BOUN and IMST treebanks for Turkish. 


## Treebanks
We collected 2 treebanks [IMST](https://github.com/UniversalDependencies/UD_Turkish-IMST) and [BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN) from their Github repos. Then we converted conllu format to json lines. The converter script can be found under `helpers/conllu_to_hf.py`. Exact instance format can be found under our HF repo documentation.


## Hugging Face
Please find our datasets and their stats at our [HF repo](https://huggingface.co/datasets/turkish-nlp-suite/Treebank-Benchmarking)

## Benchmarking

Simply run `run.sh` , it will run the given model on both treebanks. We benchmarked BERTurk on both treebanks, please find the numbers on our HF repo. 
