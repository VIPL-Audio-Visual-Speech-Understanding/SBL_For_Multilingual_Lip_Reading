# SBL_For_Multilingual_Lip_Reading
Introduction
----
This is a project for multilingual lip reading with synchronous bidirectional learning. 
In this project, we implemented it with Pytorch. Our paper can be found [here](https://arxiv.org/abs/2005.03846).

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
About the phonemes for modeling in this work, we built our phonemes table based on [DaCiDian](https://github.com/aishell-foundation/DaCiDian), 
[BigCiDian](https://github.com/speechio/BigCiDian), [g2p](https://github.com/Kyubyong/g2p) and [
g2pC](https://github.com/Kyubyong/g2pC). Here, thanks for their inspiring works. 

Some codes of this respository is based on [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer) and [end-to-end-lipreading](https://github.com/mpc001/end-to-end-lipreading). 
Thanks for their inspiring works.


In this respository, we placed four directories. 

The directories called VSR_seq2seq_Transformer_with_phonemes_LRW and VSR_seq2seq_Transformer_with_phonemes_LRW1000 
refer the work that train the model LRW and LRW1000 each other with phonemes. 
```
cd VSR_seq2seq_Transformer_with_phonemes_LRW
python train.py
```
```
cd VSR_seq2seq_Transformer_with_phonemes_LRW1000
python train.py
```
The VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify refers to the work that viewing a 1500-classes classifying task. 

In SBL_MLR, for training stage, we suggest the following three stages:
* Stage 1: For accelerating the training speed, in the stage 1, we pretrained the encoder part 
(including the visual-frontend and the transformer encoder) by a 1500 classes classifying task.
```
cd VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify
python train.py
```
* Stage 2: With a pretrained encoder model by stage 1, we went on training the SBL model. 
Loading the pretrained encoder part model, and fixing it. In this stage, we mainly trained the 
SBL transformer decoder. 
```
cp -r VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify/BEST_checkpoint_only_visual_based_lrw_lrw1000_1500.tar SBL_For_Multilingual_Lip_Reading/
cd SBL_For_Multilingual_Lip_Reading
vim utils.py ## set checkpoint default to BEST_checkpoint_only_visual_based_lrw_lrw1000_1500.tar
vim transformer/transformer.py ## set p.requires_grad = False

python train.py
```
* Stage 3: 
```
Finetune the total model.
```
* Stage 4:
```
testing the model.
```

Reference
----
If this work is useful for your research, please cite our work:
```
@article{luo2020synchronous,
  title={Synchronous Bidirectional Learning for Multilingual Lip Reading},
  author={Luo, Mingshuang and Yang, Shuang and Chen, Xilin and Liu, Zitao and Shan, Shiguang},
  journal={arXiv preprint arXiv:2005.03846},
  year={2020}
}
```
