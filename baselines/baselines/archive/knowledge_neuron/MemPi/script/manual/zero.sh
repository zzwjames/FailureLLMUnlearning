md='gpt2-xl'


python downstream_manual.py --discover_method zero --model_name ${md} --n_batches 128 --do_discover
python downstream_manual.py --discover_method zero --model_name ${md} --do_test
