# !/bin/bash
# metrics: verbmem_f privleak knowmem_f knowmem_r
# corpus: news books
# algo: 'ga_klr' 'npo' 'npo_gdr' 'npo_klr' 'ga' 'tv' 'whp'(no checkpoint:news/${algo})
# quantize: 0 for False, 1 for True

export CUDA_VISIBLE_DEVICES="0"
algos=("npo_klr" "ga_klr") 
# algos=("ga_gdr") 
# "npo_klr" "ga_klr")
quantize_4bit=0
quantize_8bit=0
corpuss=("books")

for corpus in "${corpuss[@]}"; do
  for algo in "${algos[@]}"; do
    python eval.py \
      --model_dirs " "  \    # address of the unlearned model checkpoint
      --tokenizer_dir "meta-llama/Llama-2-7b-hf"  \
      --names "${algo}"  \
      --corpus ${corpus} \
      --quantize_4bit ${quantize_4bit} \
      --quantize_8bit ${quantize_8bit} \
      --metrics knowmem_f verbmem_f privleak knowmem_r \
      --out_file ".csv"  # address of the output file
  done
done
