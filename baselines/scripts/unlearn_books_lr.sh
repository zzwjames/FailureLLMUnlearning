export CUDA_VISIBLE_DEVICES="1,2,3"
CORPUS=('books')

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR='muse-bench/MUSE-Books_target'
LLAMA_DIR='meta-llama/Llama-2-7b-hf'

MAX_LEN=2048
EPOCHS=1
# LR=('2e-5' '4e-5' '6e-5' '8e-5' '1e-4')
LR=('1e-4')
alphas=('2')
PER_DEVICE_BATCH_SIZE=4 # 8 GPUs
FT_EPOCHS=10
FT_LR='1e-5'
algos=("npo_klr" "ga_klr")
#  "ga_gdr" "npo_gdr")

for alpha in "${alphas[@]}"; do
    for LR in "${LR[@]}"; do
        for CORPUS in "${CORPUS[@]}"; do
            for algo in "${algos[@]}"; do
                python unlearn.py \
                    --algo $algo \
                    --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
                    --data_file $FORGET --retain_data_file $RETAIN \
                    --out_dir "" \ # address of the output unlearned model checkpoint
                    --max_len $MAX_LEN --epochs $EPOCHS --lr $LR --alpha $alpha \
                    --per_device_batch_size $PER_DEVICE_BATCH_SIZE
            done
        done
    done
done