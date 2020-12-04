# !/bin/sh

python -u seq2seq.py --inter-rep2 --iters 10000 --model 'GRU' --gpu 4 --tag 5 --split 'mcd' >> out/mcd_gru_inter2.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --tag 5 --split 'mcd' >> out/mcd_gru.out &
python -u seq2seq.py --inter-rep --iters 150000 --model 'GRU' --gpu 4 --tag 5 --split 'mcd' >> out/mcd_gru_inter.out &

wait
echo Group completed.

