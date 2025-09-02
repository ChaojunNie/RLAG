INPUT_PATH=USMLEtrain_ori.jsonl
OUTPUT_PATH=USMLEtrain-CTX.json


CUDA_VISIBLE_DEVICES=2,3 python elastic+nvembed.py \
    --input_data_path $INPUT_PATH \
    --output_data_path $OUTPUT_PATH