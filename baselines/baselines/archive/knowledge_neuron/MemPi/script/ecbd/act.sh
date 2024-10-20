md=$1
ratio=0.01


python inject.py --do_discover --ratio ${ratio} --discover_method 'act' --model_name ${md}
