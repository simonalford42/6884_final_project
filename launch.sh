# !/bin/sh


python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'scan' >> out/scan_copy_out.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'length' >> out/length_copy_out.out &
wait
echo Group 1 completed.
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'jump' >> out/jump_copy_out.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'turn_left' >> out/turn_left_copy_out.out &
wait
echo Group 2 completed.
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'jump_around_right' >> out/jump_around_right_copy_out.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'around_right' >> out/around_right_copy_out.out &
wait
echo Group 3 completed.
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'mcd' >> out/mcd_copy_out.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU' --split 'opposite_right' >> out/opposite_right_copy_out.out &
wait
echo Group 4 completed.

python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'scan' >> out/scan_copy_gru_a.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'length' >> out/length_copy_gru_a.out &
wait
echo Group 1 completed.
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'jump' >> out/jump_copy_gru_a.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'turn_left' >> out/turn_left_copy_gru_a.out &
wait
echo Group 2 completed.
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'jump_around_right' >> out/jump_around_right_copy_gru_a.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'around_right' >> out/around_right_copy_gru_a.out &
wait
echo Group 3 completed.
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'mcd' >> out/mcd_copy_gru_a.out &
python -u seq2seq.py --variant 'copy_out' --iters 10000 --model 'GRU_A' --split 'opposite_right' >> out/opposite_right_copy_gru_a.out &
wait
echo Group 4 completed.



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
