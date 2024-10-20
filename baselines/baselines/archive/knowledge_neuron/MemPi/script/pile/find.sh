md=$1                                                                                                      

# loss
python find_memorized_data.py --seq_len 80 --prompt_len 32 --n_batches 30 --start 48 --end 100 --model_name ${md} --do_loss

# filter
python find_memorized_data.py --seq_len 80 --prompt_len 32 --n_batches 10 --start 0 --end 100 --model_name ${md} --log
