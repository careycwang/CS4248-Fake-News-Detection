# CS4248-Fake-News-Detection

CS4248 Group 23 Project: Combining Syntax- and Semantic-level Representations for Unreliable News Classification

## Introduction

We have researched on some models such as BERT and LSTMs which have currently been implemented. Furthermore, we’ve started exploring different feature extraction methods (on both semantics and syntax).


## Datasets

- [Labeled Unreliable News (LUN)](https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset)
- [Satirical and Legitimate News (SLN)](http://victoriarubin.fims.uwo.ca/news-verification/data-to-go/)

Please make sure your dataset is downloaded and placed as follows:
```
CS4248-Fake-News-Detection
│   README.md
│   *.py
│   
└───data
    │   balancedtest.csv
    │   fulltrain.csv
    |   test.xlsx
```

## Dependencies
You have to download the dependency packages before running the code:
```
pytorch 1.8.0
pandas
tqdm
xlrd
pytorch-pretrained-bert
```

## Instructions
To train a BERT+LSTM model, you should run the following command:
```
python bert_classifier.py --batch_size 4 --max_epochs 10 --max_seq_length 500 --max_sent_length 70 --mode 0
```
You can also download our trained models in this [Google Drive link](https://drive.google.com/drive/folders/12kBrRDdM08Hp4YCxjLcYCZjjuUiiyCx4?usp=sharing).

## Experiment Results

### In domain dev set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 67.5 | 67.5 | 67.5 | 67.4
BERT + LSTM | 78.1 | 78.1 | 78.1 | 78.0
LSTM | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM | 81.4 | 82.2 | 81.4 | 81.3
GRU | 81.4 | 82.2 | 81.4 | 81.3
BiGRU | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM + Attention | 81.4 | 82.2 | 81.4 | 81.3
BiGRU + Attention | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM + Attention + GCN | 85.0 | 85.9 | 85.0 | 85.1
BiGRU + Attention + GCN | 85.0 | 85.9 | 85.0 | 85.1

### Out of domain test set 1 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 67.5 | 67.5 | 67.5 | 67.4
BERT + LSTM | 78.1 | 78.1 | 78.1 | 78.0
LSTM | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM | 81.4 | 82.2 | 81.4 | 81.3
GRU | 81.4 | 82.2 | 81.4 | 81.3
BiGRU | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM + Attention | 81.4 | 82.2 | 81.4 | 81.3
BiGRU + Attention | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM + Attention + GCN | 85.0 | 85.9 | 85.0 | 85.1
BiGRU + Attention + GCN | 85.0 | 85.9 | 85.0 | 85.1

### Out of domain test set 2 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 67.5 | 67.5 | 67.5 | 67.4
BERT + LSTM | 78.1 | 78.1 | 78.1 | 78.0
LSTM | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM | 81.4 | 82.2 | 81.4 | 81.3
GRU | 81.4 | 82.2 | 81.4 | 81.3
BiGRU | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM + Attention | 81.4 | 82.2 | 81.4 | 81.3
BiGRU + Attention | 81.4 | 82.2 | 81.4 | 81.3
BiLSTM + Attention + GCN | 85.0 | 85.9 | 85.0 | 85.1
BiGRU + Attention + GCN | 85.0 | 85.9 | 85.0 | 85.1

## Contributors

- [Chen Xihao](https://github.com/howtoosee)
- [Wang Changqin](https://github.com/archiewang0716)
- [Zhang Haolin](https://github.com/A0236053M)
