# SBL_For_Multilingual_Lip_Reading
Introduction
----
This is a project for multilingual lip reading with synchronous bidirectional learning. In this project, we implemented it with Pytorch. Our paper can be found [here](https://arxiv.org/abs/2005.03846).

Dependencies
----
* Python：3.6+
* Pytorch: 1.3+
* Others

Dataset
----
This project is trained on LRW (grayscale) and LRW-1000 (grayscale).

Training And Testing
----
In this respository, we placed four directories. The directories called VSR_seq2seq_Transformer_with_phonemes_LRW and VSR_seq2seq_Transformer_with_phonemes_LRW1000 refer the work that train the model LRW and LRW1000 each other with phonemes. The VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify refers to the work that viewing a 1500-classes classifying task. 

In SBL_MLR, for training stage, we suggest the following three stages:
* Stage 1:
```
 #Pretraining the visual-frontend and transformer encoder firstly.
```
* Stage 2:
* Stage 3:
