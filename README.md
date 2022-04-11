# CS4248-Fake-News-Detection

CS4248 Group 23 Project: Attention-based Graph Neural Networks for Multi-class Fake News Classification

## Introduction

We augment Long Short-Term Memory neural networks (LSTMs) to include the attention mechanism and empirically show its effectiveness. Then we propose a new model that attaches the attention-based LSTM layers as input to Graph Convolutional Networks (GCNs).


## Datasets

- [Labeled Unreliable News (LUN)](https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset)
- [Satirical and Legitimate News (SLN)](http://victoriarubin.fims.uwo.ca/news-verification/data-to-go/)

Pretrained embeddings GloVe: [Global Vectors for Word Representation (GloVe)](https://nlp.stanford.edu/projects/glove/)

Please make sure your dataset is downloaded and placed as follows:
```
CS4248-Fake-News-Detection
│   README.md
│   *.py
│   
└───src
    │   *.py

└───Feature Engineering
    │   *.py
│   
└───data
    │   balancedtest.csv
    │   fulltrain.csv
    |   test.xlsx
    |   glove.6B.300d.txt
```

## Dependencies
Python version: `3.7`

Please install dependencies by running the following command:
```
pip install -r requirements.txt
```

## Instructions
To train a CNN model, you should run the following command:
```
python main.py --batch_size 1024 --config cnn --encoder 1 --ntags 4 --mode 0
```

To train a LSTM model, you should run the following command:
```
python main.py --batch_size 1024 --config lstm --encoder 0 --ntags 4 --mode 0
```

To train a BiLSTM model, you should run the following command:
```
python main.py --batch_size 1024 --config bilstm --encoder 0 --ntags 4 --mode 0 --bidirectional
```

To train a LSTM + Attention model, you should run the following command:
```
python main.py --batch_size 512 --config lstm_att --encoder 0 --ntags 4 --mode 0 --attention
```

To train a BiLSTM + Attention model, you should run the following command:
```
python main.py --batch_size 512 --config bilstm_att --encoder 0 --ntags 4 --mode 0 --bidirectional --attention
```

To train a LSTM + Attention + GCN model, you should run the following command:
```
python main.py --batch_size 32 --max_epochs 10 --config lstm_att_gcn --max_sent_len 50 --encoder 2 --ntags 4 --mode 0 --attention
```

To train a BiLSTM + Attention + GCN model, you should run the following command:
```
python main.py --batch_size 32 --max_epochs 10 --config bilstm_att_gcn --max_sent_len 50 --encoder 2 --ntags 4 --mode 0 --attention --bidirectional
```

To train a BERT + LSTM model, you should run the following command:
```
python bert_classifier.py --batch_size 4 --max_epochs 10 --max_seq_length 500 --max_sent_length 70 --ntags 4 --mode 0
```

If you want to use pretrained embeddings, add `--pte data/glove.6B.300d.txt --emb_dim 300` to your command to use GloVe embeddings.

You can also download our trained models in the [Google Drive link](https://drive.google.com/drive/folders/12kBrRDdM08Hp4YCxjLcYCZjjuUiiyCx4?usp=sharing).

To test the accuracy of the models, run the following command:

Evaluate the Logistic Regression model on in-domain dev set:
```
python traditional.py lr val tfidf
```

Evaluate the Logistic Regression model on out-of-domain test set:
```
python traditional.py lr test tfidf
```

Evaluate the CNN model:
```
python main.py --batch_size 1024 --encoder 1 --model_file model_cnn.t7 --ntags 4 --mode 1
```

Evaluate the LSTM model:
```
python main.py --batch_size 1024 --encoder 0 --model_file model_lstm.t7 --ntags 4 --mode 1
```

Evaluate the BiLSTM model:
```
python main.py --batch_size 1024 --encoder 0 --model_file model_bilstm.t7 --ntags 4 --mode 1 --bidirectional
```

Evaluate the LSTM + Attention model:
```
python main.py --batch_size 1024 --encoder 0 --model_file model_lstm_att.t7 --ntags 4 --mode 1 --attention
```

Evaluate the BiLSTM + Attention model:
```
python main.py --batch_size 1024 --encoder 0 --model_file model_bilstm_att.t7 --ntags 4 --mode 1 --bidirectional --attention
```

Evaluate the LSTM + Attention + GCN model:
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 2 --model_file model_lstm_att_gcn.t7 --ntags 4  --mode 1 --attention
```

Evaluate the BiLSTM + Attention + GCN model:
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 2 --model_file model_bilstm_att_gcn.t7 --ntags 4  --mode 1 --bidirectional --attention
```

Evaluate the BERT + LSTM model:
```
python bert_classifier.py --batch_size 4 --model_file model_bert.t7 --max_seq_length 500 --max_sent_length 70 --ntags 4 --mode 1
```


## Experiment Results

### For two classes Satire / Trusted

### In domain dev set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
LR  | 92.11 | 92.21 | 91.35 | 91.38 / 92.12
CNN | 96.74 | 96.76 | 96.53 | 96.64 / 96.74
BERT + LSTM | 91.72 | 92.74| 90.56|91.31 
LSTM | 95.63 | 95.49 | 95.55 | 95.52 / 95.63
BiLSTM | 95.82 | 95.77 | 95.63 | 95.70 / 95.82
LSTM + Attention | 96.32 | 96.09 | 96.40 | 96.24 / 96.32
BiLSTM + Attention | 97.59 | 97.62 | 97.43 | 97.52 / 97.59
LSTM + Attention + GCN | 98.71 | 98.71 | 98.64 | 98.67 / 98.71
**BiLSTM + Attention + GCN** | **98.50** | **98.62** | **98.30** | **98.45 / 98.50**

### Out of domain test set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
LR  | 82.12 | 81.95 | 81.56 | 81.13 / 81.12
CNN | 66.39 | 66.54 | 66.39 | 66.31 / 66.39
BERT + LSTM | 75.83 | 76.62| 75.83| 75.65
LSTM | 82.5 | 82.62 | 82.5 | 82.48 / 82.5
BiLSTM | 76.67 | 76.75 | 76.67 | 76.65 / 76.67
LSTM + Attention | 84.17 | 84.25 | 84.17 | 84.16 / 84.17
BiLSTM + Attention | 68.05 | 72.93 | 68.05 | 66.26 / 68.05
LSTM + Attention + GCN | 87.78 | 88.58 | 87.78 | 87.71 / 87.78
**BiLSTM + Attention + GCN** | **91.11** | **91.19** | **91.11** | **91.11 / 91.11**


### For four classes Satire, Hoax, Propaganda and Trusted

### In domain dev set accuracy (train.csv 80:20 split)
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
LR  | 92.1 | 92.1 | 91.5 | 91.8 / 92.1
CNN | 96.2 | 96.3 | 95.8 | 96.1 / 96.2
BERT + LSTM | 95.3 | 95.3 | 94.7 | 95.0 / 95.3
LSTM | 88.3 | 87.8 | 87.8 | 87.7 / 88.3
BiLSTM | 93.9 | 93.7 | 93.7 | 93.7 / 93.9
LSTM + Attention | 93.3 | 93.3 | 92.4 | 92.5 / 93.3
BiLSTM + Attention | 95.8 | 95.6 | 95.6 | 95.6 / 95.8
LSTM + Attention + GCN | 98.2 | 98.1 | 98.1 | 98.1 / 98.2
**BiLSTM + Attention + GCN** | **98.3** | **98.3** | **98.2** | **98.2 / 98.3**


### Out of domain test set accuracy (balancedtest.csv)
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
LR | 59.5 | 61.5 | 59.5 | 58.4 / 59.5
CNN | 45.6 | 47.4 | 45.6 | 44.4 / 45.6
BERT + LSTM | 56.0 | 57.0 | 56.0 | 55.1 / 56.0
LSTM | 51.6 | 55.6 | 51.6 | 48.8 / 51.6
BiLSTM | 56.9 | 58.9 | 56.9 | 56.0 / 56.9
LSTM + Attention | 62.1 | 63.8 | 62.1 | 61.8 / 62.1
BiLSTM + Attention | 62.1 | 62.5 | 62.1 | 61.5 / 62.1
**LSTM + Attention + GCN** | **66.8** | **68.6** | **66.8** | **66.1 / 66.9**
BiLSTM + Attention + GCN | 57.8 | 60.7 | 57.6 | 56.4 / 57.8

## Contributors

- [Chen Xihao](https://github.com/howtoosee)
- [Wang Changqin](https://github.com/archiewang0716)
- [Zhang Haolin](https://github.com/A0236053M)
- [Zhang Lei](https://github.com/AronnZzz)
- [Hon Jia Jing](https://github.com/JiaJingHon)
