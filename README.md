# HINormer: Representation Learning On Heterogeneous Information Networks with Graph Transformer

We provide the implementaion of HINormer based on the official PyTorch implementation of HGB(https://github.com/THUDM/HGB)

## 1. Descriptions
The repository is organised as follows:

- dataset/: the original data of four benchmark dataset.
- run.py: multi-class node classificaiton of HINormer.
- run_multi.py: multi-label node classification of HINormer on IMDB.
- model.py: implementation of HINormer.
- utils/: contains tool functions.
- HGB-output/: contains test files on HGB.


## 2. Requirements

- Python==3.9.0
- Pytorch==1.12.0
- Networkx==2.8.4
- numpy==1.22.3
- dgl==0.9.0
- scikit-learn==1.1.1
- scipy==1.7.3

## 3. Running experiments

We train our model using NVIDIA TITAN Xp GPU with CUDA 10.2.

For node classification with offline evaluation:

- python run.py --dataset DBLP --len-seq 50 --dropout 0.5 --beta 0.1 --temperature 2
- python run_multi.py --dataset IMDB --len-seq 20 --beta 0.1 --temperature 0.1
- python run.py --dataset Freebase --num-gnns 3 --len-seq 30 --num-layers 3 --dropout 0 --beta 1 --temperature 0.2
- python run.py --dataset AMiner --num-gnns 3 --len-seq 100 --beta 0.1 --temperature 0.1

For node classification with online evaluation on HGB:

- python run.py --dataset DBLP-HGB --len-seq 50 --num-heads 2 --dropout 0.5 --beta 0.1 --temperature 0.1 --mode 1
- python run_multi.py --dataset IMDB-HGB --len-seq 150 --beta 0.5 --temperature 1 --mode 1

And we provide our test files on DBLP-HGB and IMDB-HGB in 'HGB-output/'.

For reproducing our results in the paper and applying HINormer on other datasets, you need to tune the values of key parameters like 'num-gnns','num-layers','len-seq', 'dropout', 'temperature' and 'beta'  in your experimental environment. 
## 4. Citation
