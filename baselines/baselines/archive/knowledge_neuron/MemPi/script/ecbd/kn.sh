md=$1
ratio=0.001

if [ ${md} == 'gpt2' ]
then
    n=1
elif [ ${md} == 'gpt2-xl' ]
then
    n=2
elif [ ${md} == 'EleutherAI/pythia-2.8b-deduped-v0' ]
then
    n=5
elif [ ${md} == 'EleutherAI/pythia-6.9b-deduped' ]
then
    n=3
fi


python inject.py --do_discover --ratio ${ratio} --discover_method 'kn' --model_name ${md} --ig_steps 20 --n_batches $n
