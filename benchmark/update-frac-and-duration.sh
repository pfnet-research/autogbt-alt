#!/bin/bash
set -euo pipefail

while getopts i: OPT
do
    case $OPT in
        i)
            INPUT=$OPTARG
            ;;
        *)
            echo "Usage: $0 -i input" 1>&2
            exit 1
            ;;
    esac
done
DATASET="$(basename $(dirname $INPUT))"
IMG="${INPUT%%/}/frac-and-n_trials.png"
DST="../assets/param-$DATASET.png"

python ./vis_frac_and_duration.py -i $INPUT
cp $IMG $DST
