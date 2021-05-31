# TodKat
Transformer encoder-decoder for emotion detection in dialogues

## Quick Start:
------------------------------------------------------
To run the model on test sets of four datasets,
1. Download the pre-trained models [save.zip](https://todcatsaves.s3.eu-west-2.amazonaws.com/save.zip) , unzip the zip file to ./TodKat/save
2. Locate to the ./TodKat/src and execute the following command in terminal:
    `python TodKat_dd.py` for DailyDialogue.
    `python TodKat_emory.py` for EmoryNLP.
    `python TodKat_iemocap.py` for IEMOCAP.
    `python TodKat_meld.py` for MELD. 

Here comes the detailed instructions on how to perform training, testing and prediction.

## Required Packages: 
------------------------------------------------------
torch==1.7.0
transformers==4.0.0
csv
typing
numpy
tqdm
pandas
ipython
ftfy==5.1

## Run on GPU:
------------------------------------------------------
Model runs on GPU by default with cuda:0. Please change device if going to train on CPU. Saved models are trained with cuda:0, change the configurations in the saved model if you want to run them on another device.

## Training:
------------------------------------------------------
Uncomment `TodKat_dd` line 613 - line 705 to train model on DailyDialog.
Uncomment `TodKat_emory` line 613 - line 705 to train model on EmoryNLP.
Uncomment `TodKat_iemocap` line 614 - line 708 to train model on IEMOCAP.
Uncomment `TodKat_meld` line 614 - line 708 to train model on MELD.

## Evaluation:
------------------------------------------------------
Comment `TodKat_dd` line 613 - line 705 and uncomment line 707 - 738 on DailyDialog.
Comment `TodKat_emory` line 613 - line 705 and uncomment line 707 - 744 on EmoryNLP.
Comment `TodKat_iemocap` line 614 - line 708 and uncomment line 710 - 741 on IEMOCAP.
Comment `TodKat_meld` line 614 - line 708 and uncomment line 710 - 742 on MELD.

## Prediction:
------------------------------------------------------
Comment `TodKat_dd` line 613 - line 705 and uncomment line 707 - 742 on DailyDialog.
Comment `TodKat_emory` line 613 - line 705 and uncomment line 707 - 748 on EmoryNLP.
Comment `TodKat_iemocap` line 614 - line 708 and uncomment line 710 - 745 on IEMOCAP.
Comment `TodKat_meld` line 614 - line 708 and uncomment line 710 - 746 on MELD.
