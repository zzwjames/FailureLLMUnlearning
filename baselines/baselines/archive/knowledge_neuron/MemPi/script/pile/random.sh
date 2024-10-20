md=$1


python downstream.py --discover_method random --model_name ${md} --n_batches 10 --do_test
