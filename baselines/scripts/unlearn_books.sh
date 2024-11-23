# algos available choice=("npo_gdr" "npo_klr" "ga_gdr" "ga" "ga_klr" "npo" "npo_gdr_sure" "npo_klr_sure" "ga_gdr_sure" "ga_klr_sure")
# alpha is for utility constraint, threshold is for filtering out salient modules
export CUDA_VISIBLE_DEVICES="0,1,2,3"
CORPUS=('books')
FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"
TARGET_DIR='muse-bench/MUSE-Books_target'
LLAMA_DIR='meta-llama/Llama-2-7b-hf'
MAX_LEN=2048
EPOCHS=1
LR='1e-5'
PER_DEVICE_BATCH_SIZE=1
algos=("rmu")

for algo in "${algos[@]}"; do
    python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "/data/zhiwei" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --alpha 1 --threshold 90 \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE
done