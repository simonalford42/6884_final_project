import random
from utils import *

import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pdb

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

# use full data for determining input/output language, in case splits somehow change it
full_data = "scan/SCAN-master/tasks.txt"
INPUT_LANG, OUTPUT_LANG, full_pairs = prepareData('scan_in', 'scan_out', full_data, False)
# used for train as well as test splits
# I add one for the <EOS> tag. Not sure if necessary but can't hurt.
MAX_LENGTH = max(len(pair[1].split(' ')) for pair in full_pairs) + 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Torch device: {}'.format(DEVICE))

"""
Helpers
"""

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs)) 

# Get Sentences
def indicesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indices = indicesFromSentence(lang, sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)


"""
Model Architecture

"""


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, dropout=.5):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # pdb.set_trace()
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout=.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output = self.dropout(self.embedding(input).view(1, 1, -1))
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=.5):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # pdb.set_trace()
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        output = embedded
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        return output, (hidden, cell)

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout=.5):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        output = self.dropout(self.embedding(input).view(1, 1, -1))
        output = F.relu(output)
        output, (hidden, cell) = self.rnn(output, (hidden, cell))
        output = self.out(output[0])
        return output, (hidden, cell)

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


"""
Training 
"""

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
        decoder_optimizer, criterion, device, mode):

    gradient_clip = 5
    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        if mode == "LSTM":
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], *encoder_hidden)
        else:
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if mode == "LSTM":
                decoder_output, decoder_hidden = decoder(
                    decoder_input, *decoder_hidden)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if mode == "LSTM":
                decoder_output, decoder_hidden = decoder(
                    decoder_input, *decoder_hidden)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clip)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length




def trainIters(encoder, decoder, pairs, n_iters, device, print_every=1000, plot_every=100,
        learning_rate=0.001, mode="GRU"):
    print('Starting training')
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(INPUT_LANG, OUTPUT_LANG, random.choice(pairs), device)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    evaluateRandomly(encoder, decoder, pairs, device, n = 100)

    for iter in range(1, n_iters + 1):
        # print("iter")
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,
                     device, mode=mode)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Duration (Remaining): %s Iters: (%d %d%%) Loss avg: %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            evaluateRandomly(encoder, decoder, pairs, device, n = 100)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # showPlot(plot_losses)
    return plot_losses


"""
Evaluation
"""


def evaluate(encoder, decoder, sentence, device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(INPUT_LANG, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(OUTPUT_LANG.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateTestSet(encoder, decoder, pairs, device):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        hits = 0
        for pair in pairs:
            output_words = evaluate(encoder, decoder, pair[0], device)
            output_sentence = ' '.join(output_words)
            if output_words[-1] == '<EOS>':
                output_sentence = ' '.join(output_words[:-1])
                if pair[1] == output_sentence:
                    hits += 1
            else:
                assert len(output_words) == MAX_LENGTH, str.format(
                        'unexpected length: {} but max is {}',
                        len(output_words), MAX_LENGTH)

        print('Evaluation accuracy: {}/{} = {:.2f}%'.format(hits, len(pairs),
            hits/len(pairs)))

    encoder.train()
    decoder.train()

    return hits/len(pairs)



def evaluateRandomly(encoder, decoder, pairs, device, n=10, verbose=False):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        hits = 0
        for i in range(n):
            pair = random.choice(pairs)
            if verbose:
                print('>', pair[0])
                print('=', pair[1])
            output_words = evaluate(encoder, decoder, pair[0], device)
            output_sentence = ' '.join(output_words)
            if verbose:
                print('<', output_sentence)
            if output_words[-1] == '<EOS>':
                output_sentence = ' '.join(output_words[:-1])
                if pair[1] == output_sentence:
                    hits += 1
            if verbose:
                print('')

    encoder.train()
    decoder.train()

    print('Hits {}/{} test samples'.format(hits, n))


def saveModel(encoder, decoder, checkpoint, path):
    checkpoint['encoder_state_dict'] = encoder.state_dict()
    checkpoint['decoder_state_dict'] = decoder.state_dict()
    torch.save(checkpoint, path)


def loadParameters(encoder, decoder, path):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return checkpoint


def scanData(path):
    _, _, pairs = prepareData('scan_in', 'scan_out', path, False)
    print('Loaded data from {}'.format(path))
    print('{} examples. Sample pair: {}'.format(len(pairs), random.choice(pairs)))
    return pairs


def trainTestSplit(train_path, test_path, save_path, device, iters=100000):
    train_path = 'scan/SCAN-master/' + train_path
    test_path = 'scan/SCAN-master/' + test_path
    train_pairs = scanData(train_path)
    test_pairs = scanData(test_path)

    input_size = INPUT_LANG.n_words
    hidden_size = 200
    output_size = OUTPUT_LANG.n_words

    encoder = EncoderRNN(input_size, hidden_size, device).to(device)
    decoder = DecoderRNN(hidden_size, output_size, device).to(device)

    train_losses = trainIters(encoder, decoder, train_pairs, iters, device)
    print('Evaluating training split accuracy')
    train_acc = evaluateTestSet(encoder, decoder, train_pairs, device)
    print('Evaluating test split accuracy')
    test_acc = evaluateTestSet(encoder, decoder, test_pairs, device)

    checkpoint = {'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_losses': train_losses}

    saveModel(encoder, decoder, checkpoint, save_path)


def loadModel(path, device):

    input_size = INPUT_LANG.n_words
    hidden_size = 200
    output_size = OUTPUT_LANG.n_words

    encoder = EncoderRNN(input_size, hidden_size, device).to(device)
    decoder = DecoderRNN(hidden_size, output_size, device).to(device)

    checkpoint = loadParameters(encoder, decoder, path)
    return encoder, decoder, checkpoint


def evalSplit(encoder, decoder, split_path, device):
    split_path = 'scan/SCAN-master/' + split_path
    split_pairs = scanData(split_path)
    pairs = scanData(split_path)
    accuracy = evaluateTestSet(encoder, decoder, pairs, device)
    return accuracy


if __name__ == '__main__':
    train_path = 'simple_split/tasks_train_simple.txt'
    test_path = 'simple_split/tasks_test_simple.txt'
    trainTestSplit(train_path, test_path, 'simple_split2.pt', device=DEVICE)

    # encoder, decoder, checkpoint = loadModel('simple_split1.pt', DEVICE)
    # print('Evaluating with loaded model')
    # evalSplit(encoder, decoder, test_path, DEVICE)

