# !/bin/sh

python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'mcd' >> out/mcd_gru_a_inter.out &
python -u seq2seq.py --model 'GRU_A' --iters 150000 --split 'mcd' >> out/mcd_gru_a.out &
wait
echo Group completed.

# python -u seq2seq.py --model 'GRU_A' --iters 150000 --split 'turn_left' >> out/turn_left_gru_a.out &
# python -u seq2seq.py --model 'GRU_A' --iters 150000 --split 'scan' >> out/scan_gru_a.out &
# python -u seq2seq.py --model 'GRU_A' --iters 150000 --split 'jump_around_right' >> out/jump_around_right_gru_a.out &
# python -u seq2seq.py --model 'GRU_A' --iters 150000 --split 'around_right' >> out/around_right_gru_a.out &
# wait
# echo Group completed.

# python -u seq2seq.py --model 'GRU_A' --iters 150000 --split 'opposite_right' >> out/opposite_right_gru_a.out &
# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'length' >> out/length_gru_a_inter.out &
# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'jump' >> out/jump_gru_a_inter.out &
# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'turn_left' >> out/turn_left_gru_a_inter.out &
# wait
# echo Group completed.

# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'scan' >> out/scan_gru_a_inter.out &
# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'jump_around_right' >> out/jump_around_right_gru_a_inter.out &
# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'around_right' >> out/around_right_gru_a_inter.out &
# python -u seq2seq.py --model 'GRU_A' --inter-rep --iters 15000 --split 'opposite_right' >> out/opposite_right_gru_a_inter.out &
# wait
# echo Group completed.
