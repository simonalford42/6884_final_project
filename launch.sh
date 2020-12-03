#!/bin/sh
python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 1 --inter-rep2 --tag 2 --split 'scan' >> out/scan_gru_inter1.out &
python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 1 --inter-rep2 --tag 2 --split 'jump' >> out/jump_gru_inter1.out &

wait
echo First group completed.

python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 2 --inter-rep2 --tag 2 --split 'turn_left' >> out/turn_left_gru_inter1.out &
python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 2 --inter-rep2 --tag 2 --split 'jump_around_right' >> out/jump_around_right_gru_inter1.out &

wait
echo Second group completed.

python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 3 --inter-rep2 --tag 2 --split 'around_right' >> out/around_right_gru_inter1.out &
python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 3 --inter-rep2 --tag 2 --split 'opposite_right' >> out/opposite_right_gru_inter1.out &

wait
echo Third group completed.

python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 4 --inter-rep2 --tag 2 --split 'length' >> out/length_gru_inter1.out &
python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 4 --inter-rep2 --tag 2 --split 'mcd' >> out/mcd_gru_inter1.out &

wait
echo Fourth group completed.

# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 1 --tag 2 --split 'scan' >> out/scan_gru1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 1 --tag 2 --split 'jump' >> out/jump_gru1.out &

# wait
# echo Fifth group completed.

# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 2 --tag 2 --split 'turn_left' >> out/turn_left_gru1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 2 --tag 2 --split 'jump_around_right' >> out/jump_around_right_gru1.out &

# wait
# echo Fifth group completed.

# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 3 --tag 2 --split 'around_right' >> out/around_right_gru1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 3 --tag 2 --split 'opposite_right' >> out/opposite_right_gru1.out &

# wait
# echo Sixth group completed.

# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 4 --tag 2 --split 'length' >> out/length_gru1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU' --gpu 4 --tag 2 --split 'mcd' >> out/mcd_gru1.out &

# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 1 --inter-rep2 --tag 2 --split 'scan' >> out/scan_gru_a_inter1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 1 --inter-rep2 --tag 2 --split 'jump' >> out/jump_gru_a_inter1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 2 --inter-rep2 --tag 2 --split 'turn_left' >> out/turn_left_gru_a_inter &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 2 --inter-rep2 --tag 2 --split 'jump_around_right' >> out/jump_around_right_gru_a_inter &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 3 --inter-rep2 --tag 2 --split 'around_right' >> out/around_right_gru_a_inter1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 3 --inter-rep2 --tag 2 --split 'opposite_right' >> out/opposite_right_gru_a_inter1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 4 --inter-rep2 --tag 2 --split 'length' >> out/length_gru_a_inter1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 4 --inter-rep2 --tag 2 --split 'mcd' >> out/mcd_gru_a_inter1.out &

# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 1 --tag 2 --split 'scan' >> out/scan_gru_a1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 1 --tag 2 --split 'jump' >> out/jump_gru_a1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 2 --tag 2 --split 'turn_left' >> out/turn_left_gru_a &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 2 --tag 2 --split 'jump_around_right' >> out/jump_around_right_gru_a &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 3 --tag 2 --split 'around_right' >> out/around_right_gru_a1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 3 --tag 2 --split 'opposite_right' >> out/opposite_right_gru_a1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 4 --tag 2 --split 'length' >> out/length_gru_a1.out &
# python -u seq2seq.py --iters 5000 --model 'GRU_A' --gpu 4 --tag 2 --split 'mcd' >> out/mcd_gru_a1.out &

