md=$1


python downstream.py --discover_method zero --model_name ${md} --n_batches 150 --do_discover
python downstream.py --discover_method zero --model_name ${md} --n_batches 10 --do_test
