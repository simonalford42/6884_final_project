#!/bin/sh
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --inter-rep --split 'scan' &>> scan_gru_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --inter-rep --split 'jump' &>> jump_gru_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --inter-rep --split 'turn_left' &>> turn_left_gru_iter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --inter-rep --split 'jump_around_right' &>> jump_around_right_gru_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --inter-rep --split 'around_right' &>> around_right_gru_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --inter-rep --split 'opposite_right' &>> opposite_right_gru_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --inter-rep --split 'length' &>> length_gru_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --inter-rep --split 'mcd' &>> mcd_gru_inter.out &

python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --split 'scan' &>> scan_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 1 --split 'jump' &>> jump_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --split 'turn_left' &>> turn_left_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 2 --split 'jump_around_right' &>> jump_around_right_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --split 'around_right' &>> around_right_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 3 --split 'opposite_right' &>> opposite_right_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --split 'length' &>> length_gru.out &
python -u seq2seq.py --iters 150000 --model 'GRU' --gpu 4 --split 'mcd' &>> mcd_gru.out &

wait
echo First group completed.

python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --inter-rep --split 'scan' &>> scan_gru_a_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --inter-rep --split 'jump' &>> jump_gru_a_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --inter-rep --split 'turn_left' &>> turn_left_gru_a_inter &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --inter-rep --split 'jump_around_right' &>> jump_around_right_gru_a_inter &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --inter-rep --split 'around_right' &>> around_right_gru_a_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --inter-rep --split 'opposite_right' &>> opposite_right_gru_a_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --inter-rep --split 'length' &>> length_gru_a_inter.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --inter-rep --split 'mcd' &>> mcd_gru_a_inter.out &

python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --split 'scan' &>> scan_gru_a.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 1 --split 'jump' &>> jump_gru_a.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --split 'turn_left' &>> turn_left_gru_a &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 2 --split 'jump_around_right' &>> jump_around_right_gru_a &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --split 'around_right' &>> around_right_gru_a.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 3 --split 'opposite_right' &>> opposite_right_gru_a.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --split 'length' &>> length_gru_a.out &
python -u seq2seq.py --iters 150000 --model 'GRU_A' --gpu 4 --split 'mcd' &>> mcd_gru_a.out &

wait
echo Second group completed.

