#!/bin/sh
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --inter-rep --tag 1 --split 'scan' &>> scan_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --inter-rep --tag 1 --split 'jump' &>> jump_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --inter-rep --tag 1 --split 'turn_left' &>> turn_left_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --inter-rep --tag 1 --split 'jump_around_right' &>> jump_around_right_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --inter-rep --tag 1 --split 'around_right' &>> around_right_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --inter-rep --tag 1 --split 'opposite_right' &>> opposite_right_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --inter-rep --tag 1 --split 'length' &>> length_gru_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --inter-rep --tag 1 --split 'mcd' &>> mcd_gru_inter1.out &

python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --tag 1 --split 'scan' &>> scan_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --tag 1 --split 'jump' &>> jump_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --tag 1 --split 'turn_left' &>> turn_left_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --tag 1 --split 'jump_around_right' &>> jump_around_right_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --tag 1 --split 'around_right' &>> around_right_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --tag 1 --split 'opposite_right' &>> opposite_right_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --tag 1 --split 'length' &>> length_gru1.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --tag 1 --split 'mcd' &>> mcd_gru1.out &

wait
echo First group completed.

python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --inter-rep --tag 1 --split 'scan' &>> scan_gru_a_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --inter-rep --tag 1 --split 'jump' &>> jump_gru_a_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --inter-rep --tag 1 --split 'turn_left' &>> turn_left_gru_a_inter &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --inter-rep --tag 1 --split 'jump_around_right' &>> jump_around_right_gru_a_inter &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --inter-rep --tag 1 --split 'around_right' &>> around_right_gru_a_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --inter-rep --tag 1 --split 'opposite_right' &>> opposite_right_gru_a_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --inter-rep --tag 1 --split 'length' &>> length_gru_a_inter1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --inter-rep --tag 1 --split 'mcd' &>> mcd_gru_a_inter1.out &

python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --tag 1 --split 'scan' &>> scan_gru_a1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --tag 1 --split 'jump' &>> jump_gru_a1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --tag 1 --split 'turn_left' &>> turn_left_gru_a &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --tag 1 --split 'jump_around_right' &>> jump_around_right_gru_a &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --tag 1 --split 'around_right' &>> around_right_gru_a1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --tag 1 --split 'opposite_right' &>> opposite_right_gru_a1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --tag 1 --split 'length' &>> length_gru_a1.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --tag 1 --split 'mcd' &>> mcd_gru_a1.out &

wait
echo Second group completed.

