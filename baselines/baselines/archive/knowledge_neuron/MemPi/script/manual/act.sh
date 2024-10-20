md='gpt2-xl'


python downstream_manual.py --discover_method act --model_name ${md} --do_discover
python downstream_manual.py --discover_method act --model_name ${md} --do_test
