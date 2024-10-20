md=$1


python downstream.py --discover_method kn --model_name ${md} --prompt_len 32 --n_batches 3 --do_discover
python downstream.py --discover_method kn --model_name ${md} --prompt_len 32 --n_batches 10  --do_test
