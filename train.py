# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import argparse

import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau


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

DEFAULT_DATA_DIR = "data"
EXPERIMENTS_DIR = "experiments"


parser = argparse.ArgumentParser()
parser.add_argument('--with_instruction', default=False, help='Use instruction or not', type=bool)
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


print('Preparing training data generators...(it will take a moment)')
train_gen = DataGenerator(word2id, context_vocab, context_vocab, train_context_path,
                                             train_qn_path, train_ans_path, args.batch_size, graph_vocab_class,
                                             context_len=context_len, question_len=answer_len,
                                             answer_len=answer_len, discard_long=False, with_inst=args.with_instruction,
                                             use_raw_graph=use_raw_graph)

print('Preparing testing data generators...(it will take a moment)')
test_gen = DataGenerator(word2id, context_vocab, context_vocab, test_context_path,
                                             test_qn_path, test_ans_path, args.batch_size, graph_vocab_class,
                                             context_len=context_len, question_len=answer_len,
                                             answer_len=answer_len, discard_long=False, with_inst=args.with_instruction,
                                             use_raw_graph=use_raw_graph)


print('Training...', end='')

steps_per_e = 200
test_steps = np.floor(1000/args.batch_size)
num_ve_per_e = np.ceil(np.ceil(8000 / args.batch_size) / steps_per_e)
init_e = 0
total_e = init_e + args.nb_epochs * num_ve_per_e

load_m0 = False
if load_m0:
    model.load_weights(args.exp_dir + "/" + args.name + "/" + model_filename + ".h5", by_name=True)

mc = ModelCheckpoint(args.exp_dir + "/" + args.name + "/" + model_filename + ".h5",
                     monitor='loss', verbose=0,
                     save_best_only=True, save_weights_only=True,
                     mode='auto', period=1)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, cooldown=1,
                              patience=5, min_lr=0.0000001, verbose=1)

tboard = TensorBoard(log_dir=args.exp_dir + "/" + args.name + "/" + model_filename,
                     write_graph=True,
                     write_images=True)

model_hist = model.fit_generator(train_gen, steps_per_epoch=steps_per_e, epochs=total_e, initial_epoch=init_e,
                                   verbose=1, max_queue_size=20, validation_data=test_gen, validation_steps=test_steps,
                                   callbacks=[tboard, mc, reduce_lr])

