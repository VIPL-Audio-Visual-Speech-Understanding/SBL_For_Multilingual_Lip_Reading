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
```
cd VSR_visual_frontend_pretraining_on_LRW_LRW1000_classify
python train.py
```
In SBL_MLR (SBL_Multilingual_Lip_reading), for training stage, we suggest the following three stages:
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
* Stage 4: The final model is available at [GoogleDrive](https://drive.google.com/file/d/113zUIOWHCAJpQzh9S5LcSsuR8HGoRUUV/view?usp=sharing).
And copy the checkpoint to SBL_Multilingual_Lip_reading. We can test the model as follows:
```
##loading the model checkpoint
cp -r test_model_checkpoint.tar SBL_Multilingual_Lip_reading/
cd SBL_Multilingual_Lip_reading
python test.py
```
Others
----
Due to the difference in quantity and quality between the two data sets of LRW and LRW1000, we can also consider
using other training methods to train multilingual lip reading model. In general, LRW1000 is more difficult train to than 
LRW. Here, we suggest another training method as follows: 
```
Step 1: Training the model with all of LRW1000 to a certain degree of convergence.
###--->train from scratch and get the first pretrained model.
Step 2: Mix 20% of LRW data and all LRW1000 data and train the model to converge.
###--->load the first pretrained model and get the second pretrained model.
Step 3: Mix 40% of LRW data and all LRW1000 data and train the model to converge.
###--->load the second pretrained model and get the third pretrained model.
Step 4: Mix 60% of LRW data and all LRW1000 data and train the model to converge.
###--->load the third pretrained model and get the fourth pretrained model.
Step 5: Mix 80% of LRW data and all LRW1000 data and train the model to converge.
###--->load the fourth pretrained model and get the fifth pretrained model.
Step 6: Mix all LRW data and all LRW1000 data and train the model to converge.
###--->load the fifth pretrained model and get the sixth pretrained model.
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
