md=$1


python downstream.py --discover_method act --model_name ${md} --do_discover
python downstream.py --discover_method act --model_name ${md} --n_batches 10 --do_test
