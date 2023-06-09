# Protocol Selection

We replicate Protocol Selection result shown in Table 4, which is covered by inference commands using our provided model. In addition, we provide commands to train the model on your own, which should produce almost the same results shown in Table 4 in EMSAssist paper. It's highly recommend to use a performant NVIDIA GPU in this part for reducing inference/training time. We use NVIDIA A30. As what we do under other folders in this repo, we provide expected output after each command.

We provide all TFLite models in this repo and all commands to evaluate TFLite below for users to check the accuracy of the fine-tuned protocol selection TFLite models. When testing TFLite models on a server with NVIDIA GPU, it's good to set `cuda_device` as `-1` so that the TFLite test process does not occupy your NVIDIA GPU. TFLite inference engine, according to [information provided here](https://github.com/tensorflow/tensorflow/issues/34536#issuecomment-565632906), can only delegate operations to mobile GPUs. This applies to all TFLite files.

As indicated in our paper, the TFLite results are almost the same with TF results. In addition, the evaluation of TFLite can only leverages CPUs, which will take several hours. So, the focus of the artifact evaluation is on TF models on the server. It's important to note each time you switch between ALBERT and other protocol selection models, we need to regenerate the tfrecord files as ALBERT uses different tokenization approaches.

## 1 Testing (Replicate with provided models)

### 1.1 Protocol Selection on Customized Local Dataset

* MetaMap/MetaMapLite (SOTA, 11 minutes): `python match_nemsis_nemsisconcepts.py --concept_set_source both --metric cosine --input_mm_file nemsis_input_mm_concept.txt --input_mml_file nemsis_input_mml_concept.txt`

> =============== protocol_selection ====================
> 
> /home/liuyi/anaconda3/envs/xgb-gpu/lib/python3.7/site-packages/scipy/spatial/distance.py:699: RuntimeWarning: invalid value encountered in sqrt
>
>  dist = 1.0 - uv / np.sqrt(uu * vv)
>
>(cosine) input metamap topk:  0.13053887402542227 0.37417469485239235 0.49031450585033703
>
>(cosine) input metamaplite topk:  0.10407169676997095 0.26762851795289877 0.33216020392931367

* ANN One-hot encoding words (55 seconds): `python emsANN.py --test_model_path ../../model/emsANN/Fitted_Words_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_46_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type words --test_only`

> 545/545 [==============================] - 3s 5ms/step - loss: 1.0586 - top1: 0.6872 - top3: 0.9104 - top5: 0.9508
> 
> inference time of model ../../model/emsANN/Fitted_Words_Desc/0001/ on server is 0:00:00.000101
>
> This run takes 0:00:49.493647

Running TFLite on PH-1 (3 mins): `python emsANN.py --test_model_path ../../model/emsANN/Fitted_Words_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --test_file no_fitted_separated_desc_code_46_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type words --test_tflite --tflite_name ANN_Fitted_Words_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
>
> 0.6865873 0.9105437 0.9505953
>
>Here is the top1/3/5 using tf math topk:
> 
>0.6865872902022665 0.9105436809639937 0.950595323483001
>
> This run takes 0:02:52.713626

* ANN One-hot encoding codes (1.4 minutes): `python emsANN.py --test_model_path ../../model/emsANN/Fitted_Codes_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_46_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type codes --test_only`

> 545/545 [==============================] - 4s 5ms/step - loss: 1.0673 - top1: 0.6817 - top3: 0.9132 - top5: 0.9516
>
> inference time of model ../../model/emsANN/Fitted_Codes_Desc/0003/ on server is 0:00:00.000103
> 
> This run takes 0:01:18.282308

Running TFLite on PH-1 (4 minutes): `python emsANN.py --test_model_path ../../model/emsANN/Fitted_Codes_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --test_file no_fitted_separated_desc_code_46_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type codes --test_tflite --tflite_name ANN_Fitted_Codes_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
>
> 0.68176734 0.91303974 0.95168555
>
> Here is the top1/3/5 using tf math topk:
>
> 0.6817673217615837 0.9130397360493473 0.9516855544398222
>
> This run takes 0:03:54.115804

* ANN One-hot encoding tokens (3.5 minutes): `python emsANN.py --test_model_path ../../model/emsANN/Fitted_Tokens_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_46_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type tokens --test_only`

> 545/545 [==============================] - 5s 8ms/step - loss: 1.0828 - top1: 0.6809 - top3: 0.9093 - top5: 0.9501
> 
> inference time of model ../../model/emsANN/Fitted_Tokens_Desc/0001/ on server is 0:00:00.000158
> 
> This run takes 0:03:32.688118

Running TFLite on PH-1 (9 minutes): `python emsANN.py --test_model_path ../../model/emsANN/Fitted_Tokens_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --test_file no_fitted_separated_desc_code_46_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type tokens --test_tflite --tflite_name ANN_Fitted_Tokens_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
>
> 0.6804476 0.9092239 0.95005023
>
> Here is the top1/3/5 using tf math topk:
>
> 0.6804475684980634 0.9092239277004734 0.9500502080045904
>
> This run takes 0:09:00.718973

* XGBoost One-hot encoding words, lr 0.1 (20 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_46_test.txt --cuda_device 0 --feature_type words --test_model_path ../../model/emsXGBoost/Fitted_Words_Desc_Lr0.1/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
>
> top1 accuracy: 0.6948214029550998
>
> top3 accuracy: 0.9241141873475829
> 
> top5 accuracy: 0.9604360923827284
> 
> This run takes 0:00:20.664986

* XGBoost One-hot encoding codes, lr 0.1 (10 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_46_test.txt --cuda_device 0 --feature_type codes --test_model_path ../../model/emsXGBoost/Fitted_Codes_Desc_Lr0.1/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.6953952087218477
>
> top3 accuracy: 0.9215033711088797
> 
> top5 accuracy: 0.9566202840338546
>
> This run takes 0:00:03.909039

* XGBoost One-hot encoding tokens, lr 0.1 (15 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_46_test.txt --cuda_device 0 --feature_type tokens --test_model_path ../../model/emsXGBoost/Fitted_Tokens_Desc_Lr0.1/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.6921245158513843
> 
> top3 accuracy: 0.9232821689857983
>
> top5 accuracy: 0.9598909769043179
>
> This run takes 0:00:14.954746

* XGBoost One-hot encoding tokens, lr 0.05 (16 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_46_test.txt --cuda_device 0 --feature_type tokens --test_model_path ../../model/emsXGBoost/Fitted_Tokens_Desc_Lr0.05/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.6933581982498924
>
> top3 accuracy: 0.9232821689857983
>
> top5 accuracy: 0.9609238272844642
>
> This run takes 0:00:16.030952

* XGBoost One-hot encoding tokens, lr 0.01 (23 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_46_test.txt --cuda_device 0 --feature_type tokens --test_model_path ../../model/emsXGBoost/Fitted_Tokens_Desc_Lr0.01/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.6937885525749534
>
> top3 accuracy: 0.9238272844642089
> 
> top5 accuracy: 0.9606369244010903
> 
> This run takes 0:00:23.975072

<!-- ### 1.2.5 BERT Baselines on Customized Local Dataset -->

* BERT_BASE (1.2 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_BertBase4_Fitted_Desc/0002/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 60s 109ms/step - loss: 0.9710 - top1_accuracy: 0.7190 - top3_accuracy: 0.9217 - top5_accuracy: 0.9577
>
> inference time of model ../../model/emsBERT/FineTune_BertBase4_Fitted_Desc/0002/ on server is 
0:00:00.110970
>
> This run takes 0:01:12.44526

Running TFLite on PH-1 (__7.2 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_BertBase4_Fitted_Desc/0002/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite --test_tflite_model_path ../../model/export_tflite/FineTune_BertBase4_Fitted_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
>
> 0.71917945 0.9217903 0.95788264
> 
> Here is the top1/3/5 using tf math topk:                      
> 
> 0.7191794577535504 0.9217902739922537 0.9578826567207
> 
> This run takes 7:12:51.097079

* BERT_PubMed (1.2 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_PubMed2_Fitted_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 61s 109ms/step - loss: 0.9883 - top1_accuracy: 0.7206 - top3_accuracy: 0.9247 - top5_accuracy: 0.9604
>
> inference time of model ../../model/emsBERT/FineTune_PubMed2_Fitted_Desc/0003/ on server is 0:00:00.111243
>
> This run takes 0:01:12.889064

Running TFLite on PH-1 (__7.2 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_PubMed2_Fitted_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite --test_tflite_model_path ../../model/export_tflite/FineTune_PubMed2_Fitted_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
> 
> 0.72015494 0.9246019 0.9598623
> 
> Here is the top1/3/5 using tf math topk:
> 
> 0.7201549275570219 0.9246019222493186 0.9598622866159805
> 
> This run takes 7:13:23.804974

* BERT_EMS (1.5 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_Fitted_Desc/0002/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 60s 107ms/step - loss: 0.9868 - top1_accuracy: 0.7189 - top3_accuracy: 0.9193 - top5_accuracy: 0.9554
>
> inference time of model ../../model/emsBERT/FineTune_Pretrained30_Fitted_Desc/0002/ on server is 0:00:00.109457
>
> This run takes 0:01:26.34841

Running TFLite on PH-1 (__7.2 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_Fitted_Desc/0002/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite --test_tflite_model_path ../../model/export_tflite/FineTune_Pretrained30_Fitted_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
> 
> 0.71892124 0.91837615 0.95578825
> 
> Here is the top1/3/5 using tf math topk:
> 
> 0.7189212451585139 0.9183761296801033 0.95578826567207
> 
> This run takes 7:15:49.777462

* ALBERT_Base (1.05 minutes):

We need to remove the previous tfrecord files as ALBERT uses different tokenization scheme when compared to other baseline models:

`rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`

Run TF test: `python emsAlBERT.py --init_model ../../init_models/albert2/base_2/  --test_model_path ../../model/emsBERT/FineTune_AlbertBase2_Fitted_Desc/0004.h5 --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 60s 107ms/step - loss: 1.0084 - top1_accuracy: 0.7138 - top3_accuracy: 0.9180 - top5_accuracy: 0.9563
> 
> inference time of model ../../model/emsBERT/FineTune_AlbertBase2_Fitted_Desc/0004.h5 on server is 0:00:00.110427
>
> This run takes 0:01:04.504529

Run TFLite test(__7 hours__): `python emsAlBERT.py --init_model ../../init_models/albert2/base_2/  --test_model_path ../../model/emsBERT/FineTune_AlbertBase2_Fitted_Desc/0004.h5 --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite --tflite_name FineTune_AlbertBase2_Fitted_Desc_batch1.tflite`

* ALBERT_Large (1.05 minutes):

We need to remove the previous tfrecord files as ALBERT uses different tokenization scheme when compared to other baseline models: 

`rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`

Run TF test: `python emsAlBERT.py --init_model ../../init_models/albert2/large_2  --test_model_path ../../model/emsBERT/FineTune_AlbertLarge2_Fitted_Desc/0003.h5 --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 162s 293ms/step - loss: 0.9987 - top1_accuracy: 0.7138 - top3_accuracy: 0.9205 - top5_accuracy: 0.9562
> 
> inference time of model ../../model/emsBERT/FineTune_AlbertLarge2_Fitted_Desc/0003.h5 on server is 0:00:00.298006
>
> This run takes 0:02:47.397797

Run TFLite test (__1 day 7 hours__): `python emsAlBERT.py --init_model ../../init_models/albert2/large_2/  --test_model_path ../../model/emsBERT/FineTune_AlbertBase2_Fitted_Desc/0003.h5 --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite --tflite_name FineTune_AlbertLarge2_Fitted_Desc_batch1.tflite`

* EMSMobileBERT (ours)

Run TF test (1.2 mins): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --do_test`

> 545/545 [==============================] - 37s 62ms/step - loss: 0.9696 - top1_accuracy: 0.7226 - top3_accuracy: 0.9270 - top5_accuracy: 0.9629
>
> inference time of model ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004/ on server is 0:00:00.067642
>
> This run takes 0:01:12.467527


Run TFLite test (__7 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004/ --test_tflite_model_path ../../model/export_tflite/FineTune_MobileEnUncase1_Fitted_Desc_batch1.tflite --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --test_tflite`

> tflite inference time of model ../../model/export_tflite/FineTune_Pretrained30_Fitted_Desc_batch1.tflite on server is 0:00:00.740871
> 
> Here is the top1/3/5 using tf sparse topk:
>
> 0.71892124 0.91837615 0.95578825
> 
> Here is the top1/3/5 using tf math topk:
>
> 0.7189212451585139 0.9183761296801033 0.95578826567207
>
> This run takes 7:10:32.027748

### 1.2 Protocol Selection on Nation-wide dataset

We need to remove the previous tfrecord files as ALBERT uses different tokenization scheme when compared to other baseline models:

`rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`

* ANN One-hot encoding words (3 mins): `python emsANN.py --test_model_path ../../model/emsANN/NoFitted_Words_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_102_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type words --test_only`

> 2117/2117 [==============================] - 11s 5ms/step - loss: 1.5312 - top1: 0.5397 - top3: 0.8322 - top5: 0.9136
>
> inference time of model ../../model/emsANN/NoFitted_Words_Desc/0001/ on server is 0:00:00.000080
>
> This run takes 0:03:16.217035

Running TFLite on PH-1 (__30 mins__): `python emsANN.py --test_model_path ../../model/emsANN/NoFitted_Words_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_102_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type words --test_tflite --tflite_name ANN_NoFitted_Words_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
>
> 0.5384746 0.831831 0.9135124
>
> Here is the top1/3/5 using tf math topk:
>
> 0.5384746000457723 0.8318309671989547 0.9135124360479281
>
> This run takes 0:32:29.080218

* ANN One-hot encoding codes (5 minutes): `python emsANN.py --test_model_path ../../model/emsANN/NoFitted_Codes_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_102_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type codes --test_only`

> 2117/2117 [==============================] - 12s 5ms/step - loss: 1.5021 - top1: 0.5432 - top3: 0.8388 - top5: 0.9166
> 
> inference time of model ../../model/emsANN/NoFitted_Codes_Desc/0003/ on server is 0:00:00.000087
> 
> This run takes 0:04:54.284115

Running TFLite on PH-1 (__38 minutes__): `python emsANN.py --test_model_path ../../model/emsANN/NoFitted_Codes_Desc/0003/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --test_file no_fitted_separated_desc_code_102_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type codes --test_tflite --tflite_name ANN_NoFitted_Codes_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
> 
> 0.542476 0.8383572 0.9159118
>
> Here is the top1/3/5 using tf math topk:
>
> 0.5424759879810709 0.8383572161561574 0.9159117922821938
> 
> This run takes 0:37:58.676564

* ANN One-hot encoding tokens (14 minutes): `python emsANN.py --test_model_path ../../model/emsANN/NoFitted_Tokens_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --test_file no_fitted_separated_desc_code_102_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type tokens --test_only`

> 2117/2117 [==============================] - 17s 8ms/step - loss: 1.5507 - top1: 0.5350 - top3: 0.8274 - top5: 0.9102
> 
> inference time of model ../../model/emsANN/NoFitted_Tokens_Desc/0001/ on server is 0:00:00.000124
>
> This run takes 0:13:45.737043

Running TFLite on PH-1 (__40 minutes__): `python emsANN.py --test_model_path ../../model/emsANN/NoFitted_Tokens_Desc/0001/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --test_file no_fitted_separated_desc_code_102_test.txt --vocab_file ../../model/emsANN/vocab.txt --test_batch_size 64 --feature_type tokens --test_tflite --tflite_name ANN_NoFitted_Tokens_Desc_batch1.tflite`

<!-- ### 1.2.8 XGBoost Baselines on Nation-wide dataset -->

* XGBoost One-hot encoding words, lr 0.1 (30 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_102_test.txt --cuda_device 1 --feature_type words --test_model_path ../../model/emsXGBoost/NoFitted_Words_Desc_Lr0.1/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.5570788391545407
> 
> top3 accuracy: 0.8461311303551785
> 
> top5 accuracy: 0.923936716056492
>
> This run takes 0:00:23.814349

* XGBoost One-hot encoding codes, lr 0.1 (20 seconds): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_102_test.txt --cuda_device 1 --feature_type codes --test_model_path ../../model/emsXGBoost/NoFitted_Codes_Desc_Lr0.1/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.5596701438875477
> 
> top3 accuracy: 0.8468620111773087
> 
> top5 accuracy: 0.9243944393996442
> 
> This run takes 0:00:21.046161

* XGBoost One-hot encoding tokens, lr 0.1 (1 minute): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_102_test.txt --cuda_device 1 --feature_type tokens --test_model_path ../../model/emsXGBoost/NoFitted_Tokens_Desc_Lr0.1/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
> 
> top1 accuracy: 0.5552922415893335
> 
> top3 accuracy: 0.8445512465578466
> 
> top5 accuracy: 0.9233534879257012
> 
> This run takes 0:01:01.877068

* XGBoost One-hot encoding tokens, lr 0.05 (1.2 minutes): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_102_test.txt --cuda_device 1 --feature_type tokens --test_model_path ../../model/emsXGBoost/NoFitted_Tokens_Desc_Lr0.05/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
>
> top1 accuracy: 0.5576620672853314
>
> top3 accuracy: 0.8453116579182447
>
> top5 accuracy: 0.9232870442145984
>
> This run takes 0:01:13.043966

* XGBoost One-hot encoding tokens, lr 0.01 (2.5 minutes): `python emsXGBoost.py --eval_dir ../../data/ae_text_data/ --test_file no_fitted_separated_desc_code_102_test.txt --cuda_device 1 --feature_type tokens --test_model_path ../../model/emsXGBoost/NoFitted_Tokens_Desc_Lr0.01/ --vocab_file ../../model/emsXGBoost/vocab.txt --test_only`

> evaluate on test set:
>
> top1 accuracy: 0.5587399319321092
>
> top3 accuracy: 0.8458948860490355
>
> top5 accuracy: 0.9234420795405048
>
> This run takes 0:02:46.572203

<!-- ### 1.2.9 BERT Baselines on Nation-wide dataset -->

* BERT_BASE (4 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_BertBase4_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 2 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 229s 108ms/step - loss: 1.3515 - top1_accuracy: 0.5960 - top3_accuracy: 0.8576 - top5_accuracy: 0.9292
>
> inference time of model ../../model/emsBERT/FineTune_BertBase4_NoFitted_Desc/0004/ on server is 0:00:00.108374
>
> This run takes 0:04:05.172377

Running TFLite on PH-1 (__28.5 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_BertBase4_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --test_tflite --test_tflite_model_path ../../model/export_tflite/FineTune_BertBase4_NoFitted_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
>
> 0.59601486 0.85775876 0.92867637
>
> Here is the top1/3/5 using tf math topk:
>
> 0.5960148538607487 0.8577587797981587 0.9286763674484877
>
> This run takes 1 day, 4:32:30.855136

* BERT_PubMed (4 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_PubMed2_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 229s 107ms/step - loss: 1.3383 - top1_accuracy: 0.5930 - top3_accuracy: 0.8588 - top5_accuracy: 0.9299
>
> inference time of model ../../model/emsBERT/FineTune_PubMed2_NoFitted_Desc/0004/ on server is 0:00:00.108039
>
> This run takes 0:04:02.875558

Running on PH-1 (__28.5 hours__):`python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_PubMed2_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --test_tflite --test_tflite_model_path ../../model/export_tflite/FineTune_PubMed2_NoFitted_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:                                                
> 
> 0.59381485 0.85882187 0.9301972                                                           
> 
> Here is the top1/3/5 using tf math topk:                                                  
> 
> 0.5938148287597912 0.8588218791758027 0.9301971901692838                                  
> 
> This run takes 1 day, 4:34:53.355314 

* BERT_EMS (5 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device 2 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 228s 107ms/step - loss: 1.3611 - top1_accuracy: 0.5913 - top3_accuracy: 0.8550 - top5_accuracy: 0.9274
>
> inference time of model ../../model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/0004/ on server is 0:00:00.107846
>
> This run takes 0:04:54.529066

Running TFLite on PH-1 (__28.5 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/0004/ --eval_dir ../../data/ae_text_data/ --cuda_device  -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --test_tflite --test_tflite_model_path ../../model/export_tflite/FineTune_Pretrained30_NoFitted_Desc_batch1.tflite`

> Here is the top1/3/5 using tf sparse topk:
> 
> 0.5911128 0.8543111 0.927473
> 
> Here is the top1/3/5 using tf math topk:
> 
> 0.5911127845082796 0.854311089455383 0.9274729980140713
> 
> This run takes 1 day, 4:34:58.242247


* ALBERT_Base:

We need to remove the previous tfrecord files as ALBERT uses different tokenization scheme: 

`rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`

Run TF test (4 mins): `python emsAlBERT.py --init_model ../../init_models/albert2/base_2/  --test_model_path ../../model/emsBERT/FineTune_AlbertBase2_NoFitted_Desc/0005.h5 --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 231s 108ms/step - loss: 1.3627 - top1_accuracy: 0.5872 - top3_accuracy: 0.8527 - top5_accuracy: 0.9260
> 
> inference time of model ../../model/emsBERT/FineTune_AlbertBase2_NoFitted_Desc/0005.h5 on server is 0:00:00.109099
> 
> This run takes 0:03:57.476973

Run TFLite test (__28 hours__): `python emsAlBERT.py --init_model /home/liuyi/emsAssist_mobisys22/init_models/albert2/base_2/  --test_model_path ../../model/emsBERT/FineTune_AlbertBase2_NoFitted_Desc/0005.h5 --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --test_tflite --tflite_name FineTune_AlbertBase2_NoFitted_Desc_batch1.tflite`

* ALBERT_Large:

We need to remove the previous tfrecord files as ALBERT uses different tokenization scheme: 

`rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`

Run TF test (11 minutes): `python emsAlBERT.py --init_model ../../init_models/albert2/large_2 --test_model_path ../../model/emsBERT/FineTune_AlbertLarge2_NoFitted_Desc/0001.h5 --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 626s 295ms/step - loss: 2.3341 - top1_accuracy: 0.5094 - top3_accuracy: 0.6429 - top5_accuracy: 0.7091
>
> inference time of model ../../model/emsBERT/FineTune_AlbertLarge2_NoFitted_Desc/0001.h5 on server is 0:00:00.295861
>
> This run takes 0:10:33.026546

Run TFLite test on PH-1 (__28 hours__): `python emsAlBERT.py --init_model ../../init_models/albert2/large_2/  --test_model_path ../../model/emsBERT/FineTune_AlbertLarge2_NoFitted_Desc/0001.h5 --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --test_tflite --tflite_name FineTune_AlbertLarge2_NoFitted_Desc_batch1.tflite`


* EMSMobileBERT (ours):

We need to remove the previous tfrecord files as ALBERT uses different tokenization scheme:
`rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`

Run TF test (5 minutes): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_MobileEnUncase1_NoFitted_Desc/0006/ --eval_dir ../../data/ae_text_data/ --cuda_device 0 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --do_test`

> 2117/2117 [==============================] - 126s 58ms/step - loss: 1.3497 - top1_accuracy: 0.5937 - top3_accuracy: 0.8599 - top5_accuracy: 0.9310
>
> inference time of model ../../model/emsBERT/FineTune_MobileEnUncase1_NoFitted_Desc/0006/ on server is 0:00:00.059323
>
> This run takes 0:04:26.477293

Running TFLite on PH-1 (__28 hours__): `python emsBERT.py --test_model_path ../../model/emsBERT/FineTune_MobileEnUncase1_NoFitted_Desc/0006/ --eval_dir ../../data/ae_text_data/ --cuda_device -1 --max_seq_len 128 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --test_tflite_model_path ../../model/export_tflite/FineTune_MobileEnUncase1_NoFitted_Desc_batch1.tflite  --test_tflite`

<!-- ### 1.2.3 Protocol Selection Testing with TensorFlowLite -->

## 2 Training

We expect the main focus of artifact evaluaiton to be part 1. We provide some commands here for users to reproduce the results in the paper by training from scratch. It's good to note the source code released in this repo is already sufficient to reproduce EMSAssist results by training from scratch.

We have been releasing the direct training commands to reduce the learning curve of users to learn using this artifact. Some of them are provided below.

Once you encounter an error that the models cannot take the data, please keep in mind to remove the tfrecord files as ALBERT uses different tokenization scheme: `rm -rf ../../data/ae_text_data/*.tfrecord ../../data/ae_text_data/*meta_data`. If you encounter additional issues during training, we welcome you to file a github issue in this repository.

### 2.1 training on customized local datasets

* ALBERT_Base: `python emsAlBERT.py --eval_dir ../data/text_data --model_dir /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_AlbertBase2_Fitted_Desc/ --init_model /home/liuyi/emsAssist_mobisys22/init_models/albert2/base_2/ --cuda_device 1 --max_seq_len 128 --train_file no_fitted_separated_desc_code_46_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_46_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --train_epoch 10 --do_train`

* ALBERT_Large: `python emsAlBERT.py --eval_dir ../data/text_data --model_dir /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_AlbertLarge2_Fitted_Desc/ --init_model /home/liuyi/emsAssist_mobisys22/init_models/albert2/large_2/ --cuda_device 1 --max_seq_len 128 --train_file no_fitted_separated_desc_code_46_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_46_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --train_epoch 10 --do_train`

* EMSMobileBERT (ours): `python emsBERT_training.py --eval_dir ../../data/text_data/ --model_dir /path/to/where/to/save/trained/models  --init_model ../../model/training_init_models/bert_en_uncased_L-12_H-768_A-12_4/ --cuda_device 0 --max_seq_len 128 --train_file no_fitted_separated_desc_code_46_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_46_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_46_test.txt --test_batch_size 64 --train_epoch 10 --do_train`

### 2.2 training on nation-wide datasets:

* ALBERT_Base: `python emsAlBERT.py --eval_dir ../data/text_data --model_dir /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_AlbertBase2_NoFitted_Desc/ --init_model /home/liuyi/emsAssist_mobisys22/init_models/albert2/base_2/ --cuda_device 2 --max_seq_len 128 --train_file no_fitted_separated_desc_code_102_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_102_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --train_epoch 10 --do_train`

* ALBERT_Large: `python emsAlBERT.py --eval_dir ../data/text_data --model_dir /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_AlbertLarge2_NoFitted_Desc/ --init_model /home/liuyi/emsAssist_mobisys22/init_models/albert2/large_2/ --cuda_device 2 --max_seq_len 128 --train_file no_fitted_separated_desc_code_102_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_102_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --train_epoch 10 --do_train`

* EMSMobileBERT (ours): `python emsBERT_training.py --eval_dir ../../data/text_data/ --model_dir /path/to/where/to/save/trained/models  --init_model ../../model/training_init_models/bert_en_uncased_L-12_H-768_A-12_4/ --cuda_device 0 --max_seq_len 128 --train_file no_fitted_separated_desc_code_102_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_102_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --train_epoch 10 --do_train`


<!-- # 2. Speech Recognition

## 2.1 

# 3. Deployment


`cd src/protocol_selection/test`
`python emsBERT.py --eval_dir ../data/text_data --model_dir /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_Pretrained30_NoFitted_Desc/ --init_model /slot1/models/official/nlp/bert/saved_models/epoch30/ --cuda_device 2 --max_seq_len 128 --train_file no_fitted_separated_desc_code_102_train.txt --train_batch_size 8 --eval_file no_fitted_separated_desc_code_102_eval.txt --eval_batch_size 64 --test_file no_fitted_separated_desc_code_102_test.txt --test_batch_size 64 --train_epoch 10 --do_test --save_tflite --tflite_name FineTune_Pretrained30_NoFitted_Desc_batch1.tflite` -->
