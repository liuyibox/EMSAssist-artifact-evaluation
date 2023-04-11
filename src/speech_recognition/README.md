# EMSConformer

`conda activate xgb-gpu`
`export PYTHONPATH=/home/liuyi/emsAssist_mobisys22/src/speech_recognition`


### Google Cloud Speech-to-Text

We do not provide the access and commands to Google Cloud Speech-to-Text service. Instead, we provide the transcription result text files for evaluation. 



### EMSConformer on Server

`conda activate xgb-gpu`

`cd ~/emsAssist_mobisys22/src/speech_recognition/examples/conformer`

`python test.py --output test_outputs/test_for_all_ae.txt --saved ~/emsAssist_mobisys22/model/speech_models/all_14.h5 --config config_PretrainLibrispeech_TrainEMS_all.yml`

> INFO:tensorflow:greedy_wer: 0.07701108604669571
>
> INFO:tensorflow:greedy_cer: 0.042713455855846405
>
> INFO:tensorflow:beamsearch_wer: 1.0
>
> INFO:tensorflow:beamsearch_cer: 1.0
>
> This run takes 0:07:28.011039

### EMSConformer on PH-1

`python evaluate_asr.py --result_file /home/liuyi/emsAssist_mobisys22/model/speech_models/all_14_tflite_test_output.tsv`

> INFO:tensorflow:wer: 0.07701108604669571
>
> INFO:tensorflow:cer: 0.04283693805336952
>
> This run takes 0:00:11.633928


To save time for artifact evaluation, here, we have pre-generated the transcripted file `all_14_tflite_test_output.tsv`. If users want to generate the transcription result file with tflite model on their own, use the command below:

`cd inference`

`python run_tflite_model_in_files_easy.py --tflite_model /home/liuyi/emsAssist_mobisys22/model/speech_models/all_14_model.tflite`

> INFO:tensorflow:wer: 0.076571024954319
> 
> INFO:tensorflow:cer: 0.04253384470939636
> 
> This run takes 0:07:19.023407