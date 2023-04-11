# End-to-End Protocol Selection


`conda activate xgb-gpu`

`cd ~/emsAssist_mobisys22/src/end2end_protocol_selection`


### Table 5: Comparing EMSAssist with SOTA (Google Cloud) on the End- to-End (E2E) protocol selection top-1/3/5 accuracy

`python emsBERT_e2e_table5.py --protocol_model /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004  --protocol_tflite_model /home/liuyi/emsAssist_mobisys22/data/text_data/emsBERT_tfrecord_files/export_tflite/FineTune_MobileEnUncase1_Fitted_Desc_batch1.tflite --cuda_device 0`

> \##### SOTA E2E Protocol Selection Accuracy #####
>
> Truth: Server [0.14, 0.35, 0.46], PH-1 [0.12, 0.27, 0.32]
>
> GC1: Server [0.14, 0.3, 0.42], PH-1 [0.11, 0.24, 0.3]
>
> GC2: Server [0.13, 0.28, 0.4], PH-1 [0.1, 0.23, 0.29]
>
> GC3: Server [0.13, 0.32, 0.43], PH-1 [0.09, 0.24, 0.29]
>
> GC4: Server [0.14, 0.32, 0.43], PH-1 [0.11, 0.26, 0.31]
>
> GC5: Server [0.12, 0.31, 0.42], PH-1 [0.09, 0.23, 0.3]
>
> GC6: Server [0.13, 0.33, 0.43], PH-1 [0.12, 0.26, 0.31]
>
> GC7: Server [0.1, 0.24, 0.34], PH-1 [0.06, 0.19, 0.26]
>
> GC8: Server [0.14, 0.32, 0.44], PH-1 [0.1, 0.26, 0.32]
>
> \##### EMSMobileBERT E2E Protocol Selection Accuracy #####
>
> Truth: Server [0.73, 0.93, 0.98], PH-1 [0.74, 0.93, 0.98]
>
> E2E: Server [0.71, 0.91, 0.95], PH-1 [0.71, 0.9, 0.95]
>
> This run takes 0:06:31.969176


### Table 6: E2E protocol selection top-1/3/5 accuracy for different users

`python emsBERT_e2e_table6.py --protocol_model /home/liuyi/emsAssist_mobisys22/model/emsBERT/FineTune_MobileEnUncase1_Fitted_Desc/0004  --protocol_tflite_model /home/liuyi/emsAssist_mobisys22/data/text_data/emsBERT_tfrecord_files/export_tflite/FineTune_MobileEnUncase1_Fitted_Desc_batch1.tflite --cuda_device 0`
