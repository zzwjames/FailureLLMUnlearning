md='gpt2-xl'

reg=30000
lr=1e-2
e=60
loss=0.1


<< 'run_all_hypers'
i=0
ds='manual'
python hyper_slim.py --model_name ${md} --stop_loss ${loss} --seed $i --epoch 1500 --dataset ${ds} --save_ckpt
python progress.py --model_name ${md} --stop_loss ${loss} --dataset ${ds} --discover_method slim --ex_i $i
python choose_downstream_hyper.py --method slim --dataset ${ds} --ex_list 0 1 2 3 4 --model_name ${md}
exit
run_all_hypers


# run a fixed hyper
python downstream_manual.py --discover_method slim --model_name ${md} --epoch $e --lr ${lr} --lambda_l1 ${reg} --stop_loss ${loss} --do_discover

python downstream_manual.py --discover_method slim --model_name ${md} --epoch $e --lr ${lr} --lambda_l1 ${reg} --stop_loss ${loss} --n_batches 10 --do_test

