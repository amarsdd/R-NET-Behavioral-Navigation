from __future__ import print_function
from collections import Counter, defaultdict
import string
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt

from compute_metrics import *


def evaluate(ground_truth, predictions):
    f1_total = em_total = 0
    total = len(ground_truth)
    err_analysis = defaultdict(list)
    assert len(ground_truth) == len(predictions)

    for i in range(total):
        truth = ground_truth[i].strip().split(' ')
        pred = predictions[i].strip().split(' ')
        f1 = f1_score(predictions[i], " ".join(truth))
        em = exact_match_score(predictions[i], " ".join(truth))
        for j in range(len(truth)):
            err_analysis[j].append(j < len(pred) and truth[j] == pred[j])
        f1_total += f1
        em_total += em
    err_dist = np.zeros([len(err_analysis)])

    for k in err_analysis:
        err_dist[k] = sum(err_analysis[k]) / float(len(err_analysis[k]))
    plt.plot(err_dist)
    plt.xlabel("pos in the answer")
    plt.ylabel("accuracy")
    plt.show()
    exact_match = 100.0 * em_total / total
    f1 = 100.0 * f1_total / total
    print('exact_match: {}, f1: {}'.format(exact_match, f1))
    return

def evaluate_new(ground_truth, predictions):
    """
    :param ground_truth: a list of strings
    :param predictions: a list of strings
    :return: nil, side effect: print out the metrics value.
    """
    assert len(ground_truth) == len(predictions)
    f1_all = 0.0
    em_all = 0.0
    ed_all = 0.0
    gem_all = 0.0
    i = 0
    for (g, p) in zip(ground_truth, predictions):
        i += 1
        # print(i)
        pred_answer = p.strip().split(" ")
        true_answer = g.strip().split(" ")
        # true_answer = [true_answer[0]] + true_answer[1::2] + [true_answer[-1]]
        f1, em, ed, gem = compute_all_metrics(pred_answer, true_answer)
        f1_all += f1
        em_all += em
        ed_all += ed
        gem_all += gem

    f1_all /= len(ground_truth)
    em_all /= len(ground_truth)
    ed_all /= len(ground_truth)
    gem_all /= len(ground_truth)
    print("f1 {}, em {}, ed {}, gem {}".format(f1_all, em_all, ed_all, gem_all))
    return


import numpy as np
import argparse

import keras

import sys
sys.setrecursionlimit(100000)

np.random.seed(10)

import numpy as np
from data_batcher_ptr import DataGenerator
from vocab import get_glove, create_vocabulary, create_vocab_class
import tensorflow as tf

import os
# os.environ["CUDA_DEVISE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_DIR = "trained_models"
DEFAULT_DATA_DIR = "data"
EXPERIMENTS_DIR = "experiments"


parser = argparse.ArgumentParser()
parser.add_argument('--with_instruction', default=False, help='Use instruction or not', type=int)
parser.add_argument('--hdim', default=100, help='Number of units in BiRNN', type=int)
parser.add_argument('--nlayers', default=3, help='Number of layers in BiRNN', type=int)
parser.add_argument('--batch_size', default=128, help='Batch size', type=int)
parser.add_argument('--nb_epochs', default=50, help='Number of Epochs', type=int)
parser.add_argument('--optimizer', default='adam', help='Optimizer', type=str)
parser.add_argument('--lr', default=None, help='Learning rate', type=float)
parser.add_argument('--dropout', default='0.0', help='Dropout', type=str)
parser.add_argument('--name', default='Rnet_navigation', help='Model dump name prefix', type=str)


parser.add_argument('--data_dir', default=DEFAULT_DATA_DIR, help='Data directory', type=str)
parser.add_argument('--exp_dir', default=EXPERIMENTS_DIR, help='Experiment results directory (Model checkpoint and Tensorboard logs)', type=str)
parser.add_argument('--model_dir', default=MODEL_DIR, help='Trained Model directory', type=str)

args = parser.parse_args()

model_filename = "model_"

if args.with_instruction:
    from model_1 import RNet
    model_filename +="1"
else:
    from model_0 import RNet
    model_filename += "0"

model_filename += "_hdim_" + str(args.hdim) + "_nlayers_" + str(args.nlayers)


use_raw_graph = False
context_len = 300
question_len = 100
answer_len = 100
embedding_size = 100
context_vocabulary_size = 200
ans_vocabulary_size = 200


# Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
train_context_path = os.path.join(args.data_dir, "train.graph")
train_qn_path = os.path.join(args.data_dir, "train.instruction")
train_ans_path = os.path.join(args.data_dir, "train.answer")
test_context_path = os.path.join(args.data_dir, "test.graph")
test_qn_path = os.path.join(args.data_dir, "test.instruction")
test_ans_path = os.path.join(args.data_dir, "test.answer")
dev_context_path = os.path.join(args.data_dir, "dev.graph")
dev_qn_path = os.path.join(args.data_dir, "dev.instruction")
dev_ans_path = os.path.join(args.data_dir, "dev.answer")

# Define path for glove vecs
glove_path = os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(embedding_size))



# Load embedding matrix and vocab mappings
emb_matrix, word2id, id2word = get_glove(glove_path, embedding_size)

# Create vocabularies of the appropriate sizes for output answer.
context_vocab_path = os.path.join(args.data_dir, "vocab%d.context" % context_vocabulary_size)

# initialize the vocabulary.
context_vocab, rev_context_vocab = create_vocabulary(context_vocab_path, train_context_path,
                                                     context_vocabulary_size)

graph_vocab_class = create_vocab_class(context_vocab)



print('Creating the model...', end='')
dimension = len(graph_vocab_class.flags) + len(graph_vocab_class.nodes) + len(graph_vocab_class.edges) + len(graph_vocab_class.nodes)

if args.with_instruction:
    model = RNet(in_dim=dimension, word2vec_dim=embedding_size, hdim=args.hdim, dropout_rate=args.dropout,
                 N=context_len, M=answer_len,
                 nB=args.nlayers, numwords=len(word2id), embedding_matrix=emb_matrix)
else:
    model = RNet(in_dim=dimension, hdim=args.hdim, dropout_rate=args.dropout, N=context_len, M=answer_len, nB=args.nlayers)

print('Done!')

print('Compiling Keras model...', end='')
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr} if args.lr else {}}
model.compile(optimizer=optimizer_config,
              loss="categorical_crossentropy", sample_weight_mode='temporal',
              metrics=['accuracy'])
print('Done!')


print('Preparing test-repeated data generators...(it will take a moment)')
test_rep = DataGenerator(word2id, context_vocab, context_vocab, test_context_path,
                                             test_qn_path, test_ans_path, args.batch_size, graph_vocab_class,
                                             context_len=context_len, question_len=answer_len,
                                             answer_len=answer_len, discard_long=False, with_inst=args.with_instruction,
                                             use_raw_graph=use_raw_graph)


print('Preparing test-new data generators...(it will take a moment)')
test_new = DataGenerator(word2id, context_vocab, context_vocab, dev_context_path,
                                             dev_qn_path, dev_ans_path, args.batch_size, graph_vocab_class,
                                             context_len=context_len, question_len=answer_len,
                                             answer_len=answer_len, discard_long=False, with_inst=args.with_instruction,
                                             use_raw_graph=use_raw_graph)


print('Evaluating...', end='')

steps_per_e = 1
test_steps = np.floor(1000/args.batch_size)
num_ve_per_e = np.ceil(test_steps / steps_per_e)
init_e = 0
total_e = init_e + num_ve_per_e

load_m0 = True
if load_m0:
    print("Loading model : "+ args.model_dir + "/" + args.name + "/" + model_filename + ".h5")
    model.load_weights(args.model_dir + "/" + args.name + "/" + model_filename + ".h5", by_name=True)


dimension = len(graph_vocab_class.flags) + len(graph_vocab_class.nodes) + len(graph_vocab_class.edges) + len(graph_vocab_class.nodes)

print("Evaluating on Test-repeated set")
gt_str = []
pred_str = []
for i in range(int(total_e)):
    for j in range(int(steps_per_e)):
        x, y, w = test_rep[j]

        preds = model.predict_on_batch(x)

        y_preds = np.argmax(preds, axis=-1)
        y_gt = np.argmax(y, axis=-1)

        gt = np.zeros([args.batch_size, answer_len, len(graph_vocab_class.edges)])
        pred = np.zeros([args.batch_size, answer_len, len(graph_vocab_class.edges)])
        for k in range(args.batch_size):
            gt[k] = x[0][k, y_gt[k], len(graph_vocab_class.flags) + len(graph_vocab_class.nodes):-len(graph_vocab_class.nodes)]
            pred[k] = x[0][k, y_preds[k],
                    len(graph_vocab_class.flags) + len(graph_vocab_class.nodes):-len(graph_vocab_class.nodes)]

        gt_b = np.argmax(gt, axis=-1)
        pred_b = np.argmax(pred, axis=-1) * w

        # gt_b = gt_b.tolist()
        # pred_b = pred_b.tolist()

        for k in range(gt_b.shape[0]):
            gt_tmp = []
            pred_tmp = []
            for l in range(answer_len):
                if w[k, l] == 1:
                    gt_tmp.append(graph_vocab_class.edges[int(gt_b[k, l])])
                    pred_tmp.append(graph_vocab_class.edges[int(pred_b[k, l])])
                else:
                    break

            pred_answer = " ".join(pred_tmp)
            true_answer = " ".join(gt_tmp[:])

            gt_str.append(true_answer)
            pred_str.append(pred_answer)

evaluate_new(gt_str, pred_str)


print("Evaluating on Test-new set")
gt_str = []
pred_str = []
for i in range(int(total_e)):
    for j in range(int(steps_per_e)):
        x, y, w = test_new[j]

        preds = model.predict_on_batch(x)

        y_preds = np.argmax(preds, axis=-1)
        y_gt = np.argmax(y, axis=-1)

        gt = np.zeros([args.batch_size, answer_len, len(graph_vocab_class.edges)])
        pred = np.zeros([args.batch_size, answer_len, len(graph_vocab_class.edges)])
        for k in range(args.batch_size):
            gt[k] = x[0][k, y_gt[k], len(graph_vocab_class.flags) + len(graph_vocab_class.nodes):-len(graph_vocab_class.nodes)]
            pred[k] = x[0][k, y_preds[k],
                    len(graph_vocab_class.flags) + len(graph_vocab_class.nodes):-len(graph_vocab_class.nodes)]

        gt_b = np.argmax(gt, axis=-1)
        pred_b = np.argmax(pred, axis=-1) * w

        # gt_b = gt_b.tolist()
        # pred_b = pred_b.tolist()

        for k in range(gt_b.shape[0]):
            gt_tmp = []
            pred_tmp = []
            for l in range(answer_len):
                if w[k, l] == 1:
                    gt_tmp.append(graph_vocab_class.edges[int(gt_b[k, l])])
                    pred_tmp.append(graph_vocab_class.edges[int(pred_b[k, l])])
                else:
                    break

            pred_answer = " ".join(pred_tmp)
            true_answer = " ".join(gt_tmp[:])

            gt_str.append(true_answer)
            pred_str.append(pred_answer)

evaluate_new(gt_str, pred_str)

print('Done!')