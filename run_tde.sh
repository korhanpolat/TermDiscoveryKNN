#!/usr/bin/env bash

TDEROOT=$1
RESPATH=$2
CORPUS=$3
UTDSYS=$4
OUTFILE=$5
JOBS=$6
CNF=$7
SOURCE=$8

echo $SOURCE

source $SOURCE

echo $RESPATH $CORPUS $UTDSYS $OUTFILE

cd $TDEROOT
conda activate TDE

python eval_sign.py $RESPATH $CORPUS $UTDSYS $OUTFILE -n $JOBS -cnf $CNF

conda deactivate
