# !/bin/sh

python -u seq2seq.py --inter-rep2 --iters 10000 --split 'mcd' >> out/mcd_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'length' >> out/length_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'jump' >> out/jump_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'turn_left' >> out/turn_left_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'jump_around_right' >> out/jump_around_right_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'around_right' >> out/around_right_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'opposite_right' >> out/opposite_right_inter2.out
python -u seq2seq.py --inter-rep2 --iters 10000 --split 'scan' >> out/scan_inter2.out
echo Group completed.

