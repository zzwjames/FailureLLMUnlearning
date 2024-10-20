md=$1
ratio=0.01


if [ ${md} == 'gpt2' ]
then
    reg=15000
    lr=5e-3
    loss=0.2
elif [ ${md} == 'gpt2-xl' ]
then
    reg=80000
    lr=1e-3
    loss=0.1
elif [ ${md} == 'EleutherAI/pythia-2.8b-deduped-v0' ]
then
    reg=30000
    lr=1e-3
    loss=0.1
elif [ ${md} == 'EleutherAI/gpt-j-6b' ]
then
    reg=15000
    lr=5e-3
    loss=0.1
elif [ ${md} == 'EleutherAI/pythia-6.9b-deduped' ]
then
    reg=10000
    lr=1e-2
    loss=0.1
fi


# run hyper tuning on 5 dev examples
for seed in {0..4}
do
    python hyper_slim.py --model_name ${md} --stop_loss ${loss} --seed ${seed} --epoch 4000 --ratio ${ratio}
done

# run a fixed hyper on all examples
python inject.py --do_discover --epoch 4000 --lr ${lr} --lambda_l1 ${reg} --discover_method 'slim' --ratio ${ratio} --model_name ${md} --stop_loss ${loss}
