md=$1
loss=0.4

if [ ${md} == 'EleutherAI/pythia-2.8b-deduped-v0' ]
then
    reg=2000
    lr=1e-3
    p=0.9
    b=0.33
    e=700
elif [ ${md} == 'EleutherAI/pythia-6.9b-deduped' ]
then
    reg=2000
    lr=1e-3
    p=0.9
    b=0.33
    e=200
fi

<< 'run_all_hypers'
i=0
ds='pile'
python hyper_HC.py --model_name ${md} --stop_loss ${loss} --seed $i --epoch 1500 --dataset ${ds} --save_ckpt
python progress.py --model_name ${md} --stop_loss ${loss} --dataset ${ds} --discover_method HC --ex_i $i
python choose_downstream_hyper.py --method HC --ex_list 0 1 2 3 4 --model_name ${md}
exit
run_all_hypers


# run a fixed hyper
python downstream.py --discover_method HC --model_name ${md} --epoch $e --lr ${lr} --lambda_l1 ${reg} --stop_loss ${loss} --mask_p $p --beta $b --prompt_len 32 --do_discover

python downstream.py --discover_method HC --model_name ${md} --epoch $e --lr ${lr} --lambda_l1 ${reg} --stop_loss ${loss} --mask_p $p --beta $b --prompt_len 32 --n_batches 10 --do_test

