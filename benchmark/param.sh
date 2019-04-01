#!/bin/bash
set -euo pipefail

while getopts d: OPT
do
    case $OPT in
        d)
            DATASET=$OPTARG
            ;;
        *)
            echo "Usage: $0 -d dataset" 1>&2
            exit 1
            ;;
    esac
done

case $DATASET in
    airline)
        TRAIN_FRACS="0.01 0.05 0.1"
        ;;
    amazon)
        TRAIN_FRACS="0.1 0.5 1.0"
        ;;
    bank)
        TRAIN_FRACS="0.1 0.5 1.0"
        ;;
    *)
        echo "Invalid Dataset"  1>&2
        exit 1
        ;;
esac

result_dir="result/$DATASET"
n_jobs=16
for seed in `seq 1 30`
do
    for n_trials in 1 10 20 30
    do
        for frac in $TRAIN_FRACS
        do
            echo "PYTHONPATH=.. python main.py --model auto --task $DATASET --model-train-frac $frac --n-trials $n_trials --result-dir $result_dir --n-jobs $n_jobs --seed $seed"
        done
    done
done
