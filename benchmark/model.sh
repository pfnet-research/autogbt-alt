#!/bin/bash

n_jobs=16
result_dir="./result/all"
for seed in `seq 1 30`
do
    for model in "auto" "xgb" "lgb"
    do
    echo "PYTHONPATH=.. python main.py --model $model --task airline --result-dir $result_dir --n-jobs $n_jobs --seed $seed"
    echo "PYTHONPATH=.. python main.py --model $model --task bank --result-dir $result_dir --n-jobs $n_jobs --seed $seed"
    echo "PYTHONPATH=.. python main.py --model $model --task amazon --result-dir $result_dir --n-jobs $n_jobs --seed $seed"
    done
    # xgb and lgb will be OOM error on 64Gi machine
    echo "PYTHONPATH=.. python main.py --model auto --task avazu --result-dir $result_dir --n-jobs $n_jobs --seed $seed"
done
