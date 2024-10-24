# set `quantize_4bit=0, quantize_8bit=0` to test model in full precision. 
# `quantize_4bit=1, quantize_8bit=0` to test model in 4-bit. 
# `quantize_4bit=0, quantize_8bit=1` to test model in 8-bit
export CUDA_VISIBLE_DEVICES="0,1,2,3"
algos=("ga_gdr") 
quantize_4bit=0
quantize_8bit=0
corpuss=("books")

for corpus in "${corpuss[@]}"; do
  for algo in "${algos[@]}"; do
    python eval.py \
      --model_dirs "muse-bench/MUSE-Books_target" \
      --tokenizer_dir "meta-llama/Llama-2-7b-hf"  \
      --names "${algo}"  \
      --corpus ${corpus} \
      --quantize_4bit ${quantize_4bit} \
      --quantize_8bit ${quantize_8bit} \
      --metrics knowmem_f verbmem_f privleak knowmem_r \
      --out_file "output.csv"
  done
done
