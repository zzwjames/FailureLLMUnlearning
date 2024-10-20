md=$1
ratio=0.001

if [ ${md} == 'gpt2' ]
then
    if [ ${ratio} == 0.01 ]
    then
        lr=1e-3
        reg=100
        p=0.5
        b=0.67
        loss=0.6
    elif [ ${ratio} == 0.001 ]
    then
        lr=1e-2
        reg=1000
        p=0.5
        b=0.33
        loss=0.4
    fi
elif [ ${md} == 'gpt2-xl' ]
then
    lr=1e-3
    reg=500
    p=0.2
    b=0.67
    loss=0.4
elif [ ${md} == 'EleutherAI/pythia-2.8b-deduped-v0' ]
then
    if [ ${ratio} == 0.01 ]
    then
        lr=1e-3
        reg=500
        p=0.2
        b=0.33
        loss=0.1
    elif [ ${ratio} == 0.001 ]
    then
        lr=1e-3
        reg=2000
        p=0.5
        b=0.33
        loss=0.3
    fi
elif [ ${md} == 'EleutherAI/gpt-j-6b' ]
then
    lr=1e-3
    reg=1000
    p=0.5
    b=0.33
    loss=0.1
elif [ ${md} == 'EleutherAI/pythia-6.9b-deduped' ]
then
    lr=5e-3
    reg=2000
    p=0.5
    b=0.33
    loss=0.1
fi

# run hyper tuning on 5 dev examples
for i in {0..4}
do
    python hyper_HC.py --model_name ${md} --stop_loss 0.1 --seed $i --epoch 4000 --dataset ecbd --ratio ${ratio}
done

# run a fixed hyper on all examples
python inject.py --do_discover --epoch 4000 --lr ${lr} --lambda_l1 ${reg} --mask_p $p --beta $b --discover_method 'HC' --ratio ${ratio} --model_name ${md} --threshold 0.1 --stop_loss ${loss}


