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
This project is trained on LRW (grayscale) and LRW-1000 (grayscale).

Training And Testing
----
About the phonemes for modeling in this work, we built our phonemes vocabulary based on [DaCiDian](https://github.com/aishell-foundation/DaCiDian), 
[BigCiDian](https://github.com/speechio/BigCiDian), [g2p](https://github.com/Kyubyong/g2p) and [
g2pC](https://github.com/Kyubyong/g2pC). Here, thanks for their inspiring works. 

Some codes of this respository is based on [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer) and [end-to-end-lipreading](https://github.com/mpc001/end-to-end-lipreading). 
Thanks to them.

In this respository, we placed four directories. 

The directories named "VSR_seq2seq_Transformer_with_phonemes_LRW" and "VSR_seq2seq_Transformer_with_phonemes_LRW1000" 
denote the work that train the model LRW and LRW1000 each other with phonemes. 
```
cd VSR_seq2seq_Transformer_with_phonemes_LRW
python train.py
```
```
cd VSR_seq2seq_Transformer_with_phonemes_LRW1000
python train.py
```
The "VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify" refers to the work which is a 1500-classes classifying task. 
```
cd VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify
python train.py
```
In SBL_MLR ("SBL_Multilingual_Lip_reading"), for training stage, we can run the codes as follows:
```
cd SBL_Multilingual_Lip_Reading/
step 1: set teach_forcing_rate=0.5--> python train.py
step 2: set teach_forcing_rate=0.1--> python train.py
```
However, the above direct method will cost us much time for coverging. 
So, here, we suggest the following three stages to accelerate the training process:
* Stage 1: We pretrained the encoder part 
(including the visual-frontend and the transformer encoder) by a 1500 classes classifying task as follows.
```
cd VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify
python train.py
```
* Stage 2: With the pretrained encoder model by stage 1 as the initialized encoder, we went on training the SBL model. 
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
* Stage 3: Based on stage 2 and stage 3, we could get a good pretrained encoder (including visual-frontend
 and transformer encoder) and a good pretrained SBL decoder. By loading the pretrained model, we set teach_forcing_rate=0.5
 and set p.requires_grad = True. By finetuning the model, we can get a good result.
```
cd SBL_Multilingual_Lip_reading/
python train.py
```
* Stage 4: After training, we can test the model as follows:
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
