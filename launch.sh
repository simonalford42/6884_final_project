# !/bin/sh

python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --split 'mcd' >> out/mcd_gru.out &

wait
echo Group completed.

