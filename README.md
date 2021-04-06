# SBL_For_Multilingual_Lip_Reading
Introduction
----
This is a project for multilingual lip reading with synchronous bidirectional learning. 
In this project, we implemented it with Pytorch. Our paper can be found [here](https://vipl.ict.ac.cn/uploadfile/upload/2020093011033041.pdf).

Dependencies
----
* Python：3.6+
* Pytorch: 1.3+
* Others: opencv, numpy, glob, editdistance and so on.

Dataset
----
This project is performed on LRW (grayscale) and [LRW-1000](https://vipl.ict.ac.cn/view_database.php?id=14) (grayscale).

Training And Testing
----
About the phonemes for modeling in this work, the phonemes vocabulary is based on [DaCiDian](https://github.com/aishell-foundation/DaCiDian), 
[BigCiDian](https://github.com/speechio/BigCiDian), [g2p](https://github.com/Kyubyong/g2p) and [
g2pC](https://github.com/Kyubyong/g2pC). Here, thanks for their inspiring works. 

Some codes of this respository is based on [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer) and [end-to-end-lipreading](https://github.com/mpc001/end-to-end-lipreading). 
Thanks to them.

There are four directories in this repository. 

The directory named "VSR_seq2seq_Transformer_with_phonemes_LRW" denotes the work that we train the model with phonemes on LRW and "VSR_seq2seq_Transformer_with_phonemes_LRW1000" 
denotes the work that we train the model with phonemes with LRW1000. 
```
cd VSR_seq2seq_Transformer_with_phonemes_LRW
python train.py
```
```
cd VSR_seq2seq_Transformer_with_phonemes_LRW1000
python train.py
```
The "VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify" refers to the work which is a 1500-classes classifying task based on the mixtures of all the word labels in LRW and LRW-1000. 
```
cd VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify
python train.py
```
In SBL_MLR ("SBL_Multilingual_Lip_reading"), for training stage, we can run the codes directly as follows:
```
cd SBL_Multilingual_Lip_Reading/
step 1: set teach_forcing_rate=0.5--> python train.py
step 2: set teach_forcing_rate=0.1--> python train.py
```
To accelerate the training process, we also suggest another training method, including the following three stages:
* Stage 1: Pretraining the encoder part 
(including the visual-frontend and the transformer encoder) by a 1500-class classification task as follows.
```
cd VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify
python train.py
```
* Stage 2: With the pretrained encoder model obtained at stage 1 as the initialized encoder, the SBL model can be trained further to learn the decoder part. 
Loading the pretrained encoder part model, and fixing it. In this stage, we mainly trained the 
SBL transformer decoder. 
```
cp -r VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify/BEST_checkpoint_only_visual_based_lrw_lrw1000_1500.tar SBL_For_Multilingual_Lip_Reading/
cd SBL_For_Multilingual_Lip_Reading
vim utils.py ## set checkpoint default to BEST_checkpoint_only_visual_based_lrw_lrw1000_1500.tar
vim transformer/transformer.py ## set p.requires_grad = False

step 1: set teach_forcing_rate=0.5--> python train.py
step 2: set teach_forcing_rate=0.1--> python train.py
```
* Stage 3: The final result can be obtained by finetuning the model based on the pretrained parameters obtained in stage 2. In this process, the teach_forcing_rate and p.requires_grad are set as 0.5 and True respectively.
```
cd SBL_Multilingual_Lip_reading/
python train.py
```
Finally, for test, the test.py could be run to obtain the testing results.
```
python test.py
```

Reference
----
If this is useful for your research, please cite our work:
```
@article{luo2020synchronous,
  title={Synchronous Bidirectional Learning for Multilingual Lip Reading},
  author={Luo, Mingshuang and Yang, Shuang and Chen, Xilin and Liu, Zitao and Shan, Shiguang},
  journal={in proceeding of British Machine Vision Conference},
  year={2020}
}
```
