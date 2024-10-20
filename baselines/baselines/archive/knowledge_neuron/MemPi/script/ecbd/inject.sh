md=$1
ratio=0.001

if [ ${ratio} == 0.01 ]
then
    lr=0.01
elif [ ${ratio} == 0.001 ]
then
    lr=0.05
fi


python inject.py --do_inject --ratio ${ratio} --model_name ${md} --lr ${lr} --epoch 200
