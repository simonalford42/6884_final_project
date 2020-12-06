# Final Project: 6.884 Neurosymbolic models for NLP

This repository contains our work for our project "We Put the Symbolic in Neurosymbolic: Tackling Compositionality by Learning Rules From Data".
The work can be broadly divided into three parts:

1. Automatically learning intermediate representations using DreamCoder.
2. Testing the benefits and theoretical limitations of intermediate
   representations via the InterSCAN variant to SCAN.
3. End-to-end symbolic learning via program synthesis.

Our experiments testing on SCAN and InterSCAN can be found in
`seq2seq.py`, `mcd.py`, `utils.py`, and `launch.sh`. The different splits can be found
in `scan/SCAN-master`. For example, `around_right_train_inter.txt` contains the training set for the around right split using intermediate representations.
Code for our experiments automatically learning intermediate representations and the end-to-end program induction approach can be found in `scan_dreamcoder/`. Note: Experiments were conducted with a different fork of ec, so files are not fully integrated with the DreamCoder code base as presented here. 

Output files from using dreamcoder are found in `scan_dc_out`. Output files from
training on SCAN and InterSCAN are in `out`. Saved models and checkpoints are in
`saved`.

