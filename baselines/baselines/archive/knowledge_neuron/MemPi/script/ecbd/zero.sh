md=$1
ratio=0.001

if [ ${md} == 'gpt2' ]
then
    n=10
elif [ ${md} == 'gpt2-xl' ]
then
    n=64
elif [ ${md} == 'EleutherAI/pythia-2.8b-deduped-v0' ]
then
    n=200
elif [ ${md} == 'EleutherAI/gpt-j-6b' ]
then
    n=128
fi

python inject.py --do_discover --ratio ${ratio} --discover_method 'zero' --model_name ${md} --n_batches $n
