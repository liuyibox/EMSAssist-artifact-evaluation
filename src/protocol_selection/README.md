# EMSAssist
This repository contains the finalized files for EMSAssist. For artifact evaluation (AE) purposes, i.e., you only want to reproduce the results with our fine-tuned TensorFlow/TensorFlowLite (TF/TFLite) models, you can just look at the testing sections. In cases where you want to develop your customized protocol selection or automatic speech recognition (ASR) models, you can refer to the training sections.

We tested this repo with TensorFlow-GPU versions: 2.9, 2.11

# 1. Protocol Selection

## 1.1 Requirements

Install the required modules for training/testing protocol selection models:

`pip install -r requirements.txt`

When testing TFLite models on a server with NVIDIA GPU, it's good to set `cuda_device` as `-1` so that the TFLite test process does not occupy your NVIDIA GPU. TFLite inference engine, according to [information provided here](https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906), can only delegate operations to mobile GPUs. We have verified the TFLite output for the same input is the same for both servers and mobile phones, so the following TFLite evaluation 

## 1.2 Testing

### 1.2.1 Protocol Selection on Customized Local Dataset

EMSMobileBERT (ours, 1.2 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 37s 62ms/step - loss: 0.9696 - top1_accuracy: 0.7226 - top3_accuracy: 0.9270 - top5_accuracy: 0.9629
> inference time of model ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004/ on server is 0:00:00.067642
> This run takes 0:01:12.467527



BERT_BASE:

TF (1.2 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_BertBase4_Fitted_Desc/0002/ --eval_dir ../../data/ae_text_data/ --cuda_device 1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 60s 109ms/step - loss: 0.9710 - top1_accuracy: 0.7190 - top3_accuracy: 0.9217 - top5_accuracy: 0.9577
> inference time of model ../../model/emsBERT/FineTune_BertBase4_Fitted_Desc/0002/ on server is 0:00:00.110970
> This run takes 0:01:12.44526

BERT_PubMed:

TF (1.2 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_PubMed2_Fitted_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device 1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 61s 109ms/step - loss: 0.9883 - top1_accuracy: 0.7206 - top3_accuracy: 0.9247 - top5_accuracy: 0.9604
> inference time of model ../../model/emsBERT/FineTune_PubMed2_Fitted_Desc/0003/ on server is 0:00:00.111243
> This run takes 0:01:12.889064

BERT_EMS:

TF (1.5 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_Fitted_Desc/0002/ --eval_dir ../../data/ae_text_data/ --cuda_device 2 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 60s 107ms/step - loss: 0.9868 - top1_accuracy: 0.7189 - top3_accuracy: 0.9193 - top5_accuracy: 0.9554
> inference time of model ../../model/emsBERT/FineTune_Pretrained30_Fitted_Desc/0002/ on server is 0:00:00.109457
> This run takes 0:01:26.34841

TFLite (): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_Fitted_Desc/0002/ --test_tflite_model_path ../../model/export_tflite/FineTune_Pretrained30_Fitted_Desc_batch1.tflite --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite`



### 1.2.2 Protocol Selection on Nation-wide dataset

EMSMobileBERT (ours): ``

> 2117/2117 [==============================] - 126s 58ms/step - loss: 1.3497 - top1_accuracy: 0.5937 - top3_accuracy: 0.8599 - top5_accuracy: 0.9310
> inference time of model ../../model/emsBERT/FineTune_MobileEnUncase1_NoFitted_Desc/0006/ on server is 0:00:00.059323
> This run takes 0:02:42.719305

BERT_BASE:

TF (4 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_BertBase4_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 2 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 229s 108ms/step - loss: 1.3515 - top1_accuracy: 0.5960 - top3_accuracy: 0.8576 - top5_accuracy: 0.9292
> inference time of model ../../model/emsBERT/FineTune_BertBase4_NoFitted_Desc/0004/ on server is 0:00:00.108374
> This run takes 0:04:05.172377

BERT_PubMed:

TF (4 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_PubMed2_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 229s 107ms/step - loss: 1.3383 - top1_accuracy: 0.5930 - top3_accuracy: 0.8588 - top5_accuracy: 0.9299
> inference time of model ../../model/emsBERT/FineTune_PubMed2_NoFitted_Desc/0004/ on server is 0:00:00.108039
> This run takes 0:04:02.875558

BERT_EMS 

TF (5 mins):

`python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 2 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 228s 107ms/step - loss: 1.3611 - top1_accuracy: 0.5913 - top3_accuracy: 0.8550 - top5_accuracy: 0.9274
> inference time of model ../../model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/0004/ on server is 0:00:00.107846
> This run takes 0:04:54.529066



TFLite (5 mins):

`python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/0004/ --test_tflite_model_path ../../model/export_tflite/FineTune_Pretrained30_NoFitted_Desc_batch1.tflite --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_tflite`


EMSMobileBERT

TF

## 1.3 Training

The BERT_Base model 

# 2. Speech Recognition

## 2.1 

# 3. Deployment




`cd src/protocol_selection/test`
`python emsBERT.py --eval_dir ../data/text_data --model_dir /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/ --init_model /slot1/models/official/nlp/bert/saved_models/epoch30/ --cuda_device 2 --max_seq_len 128 --train_file no_fitted_separated_desc_code_102_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_102_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --train_epoch 10 --do_test --save_tflite --tflite_name FineTune_Pretrained30_NoFitted_Desc_batch1.tflite`
