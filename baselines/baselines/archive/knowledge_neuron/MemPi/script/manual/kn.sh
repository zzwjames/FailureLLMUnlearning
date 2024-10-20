md='gpt2-xl'


python downstream_manual.py --discover_method kn --model_name ${md} --n_batches 3 --do_discover
python downstream_manual.py --discover_method kn --model_name ${md} --do_test
