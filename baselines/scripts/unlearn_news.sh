export CUDA_VISIBLE_DEVICES="2,3"
CORPUS=('news')

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR='muse-bench/MUSE-News_target'
LLAMA_DIR='meta-llama/Llama-2-7b-hf'

MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=2 # 8 GPUs
FT_EPOCHS=10
FT_LR='1e-5'
# algos=("npo_gdr" "npo_klr" "ga_gdr" "ga" "ga_klr" "npo")
algos=("ga_klr")

for CORPUS in "${CORPUS[@]}"; do
    for algo in "${algos[@]}"; do
        python unlearn.py \
            --algo $algo \
            --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
            --data_file $FORGET --retain_data_file $RETAIN \
            --out_dir "" \ # address of the output unlearned model checkpoint
            --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
            --per_device_batch_size $PER_DEVICE_BATCH_SIZE
    done
done


