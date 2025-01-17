"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training

Adapted from:
https: // github.com / StanfordVL / behavioral_navigation_nlp
"""

from __future__ import absolute_import
from __future__ import division

import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID, SOS_ID, EOS_ID, Vocab, create_vocab_class, instruction_tokenizer
from keras.utils import to_categorical
import keras
import itertools


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_ids, ans_mask,
                 ans_tokens, context_embeddings):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens
        self.context_embeddings = context_embeddings

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.ans_ids = ans_ids
        self.ans_mask = ans_mask
        self.ans_tokens = ans_tokens

        self.batch_size = len(self.context_tokens)


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id, is_instr=False):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    if is_instr:
        tokens = instruction_tokenizer(sentence)  # list of strings
    else:
        tokens = split_by_whitespace(sentence)

    # if simply split in tokens.
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    ''' for debugging
    if UNK_ID in ids:
        print(tokens[ids.index(UNK_ID)], " ".join(tokens))
    '''
    return tokens, ids

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch)

def reorganize(context_line, ans_line):
    start = ans_line.strip().split()[:3]
    end = ans_line.strip().split()[-3:]
    context_trip_list = context_line.strip().split(';')
    trips_contain_full_start = []
    trips_contain_full_end = []
    trips_contain_start = []
    trips_not_contain_start = []

    for trip_str in context_trip_list:
        tmp = 0
        if start[1] in trip_str:
            tsplt = trip_str.split(start[1])
            if start[0] in tsplt[0] and start[2] in tsplt[1]:
                tmp = 1

        if end[1] in trip_str:
            tsplt = trip_str.split(end[1])
            if end[0] in tsplt[0] and end[2] in tsplt[1]:
                tmp = 2


        if tmp == 1:
            trips_contain_full_start.append(trip_str)
        elif tmp == 2:
            trips_contain_full_end.append(trip_str)
        # elif start[0] in trip_str:
        # # if tmp == 1:
        #     trips_contain_start.append(trip_str)
        # else:
        #     trips_not_contain_start.append(trip_str)

        elif start[0] in trip_str:
        # if tmp == 1:
            trips_contain_start.append(trip_str)
        else:
            trips_not_contain_start.append(trip_str)
    if trips_not_contain_start[0][0] != ' ':
        trips_not_contain_start[0] = ' ' + trips_not_contain_start[0]
    if trips_contain_start[0][0] != ' ':
        trips_contain_start[0] = ' ' + trips_contain_start[0]
    if trips_contain_full_end[0][0] != ' ':
        trips_contain_full_end[0] = ' ' + trips_contain_full_end[0]
    # if trips_contain_full_start[0][0] != ' ':
    #     trips_contain_full_start[0] = ' ' + trips_contain_full_end[0]

    # # organized_context_line = ";".join(trips_contain_full_start + trips_contain_start + trips_not_contain_start).strip() + '\n'
    # organized_context_line = ";".join(trips_contain_start + trips_not_contain_start).strip() + '\n'

    if len(trips_contain_full_start)!= 1:
        trips_contain_full_start = []
        trips_contain_full_start.append(' ' + start[0] + ' ' + start[1] + ' ' + start[2] + ' ')
    if len(trips_contain_full_end) != 1:
        trips_contain_full_end = []
        trips_contain_full_end.append(' ' + end[0] + ' ' + end[1] + ' ' + end[2]+ ' ')

    # organized_context_line = ";".join(trips_contain_full_start + trips_contain_start + trips_not_contain_start).strip() + '\n'
    organized_context_line = ";".join(trips_contain_full_start + trips_contain_full_end + trips_contain_start + trips_not_contain_start).strip() + '\n'

    trips_contain_full_start = ";".join(trips_contain_full_start).strip() + '\n'
    trips_contain_full_end = ";".join(trips_contain_full_end).strip() + '\n'
    #assert len(organized_context_line) == len(context_line), "len {} {}{} len {}".\
    #      format(len(context_line), context_line, organized_context_line, len(organized_context_line))
    return organized_context_line, trips_contain_full_start, trips_contain_full_end


def refill_batches( word2id, context2id, ans2id, context_file, qn_file, ans_file, batch_size, context_len,
                   question_len, ans_len, discard_long, shuffle=True, output_goal=False):
    """
    Adds more batches into the "batches" list.
    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    print "Refilling batches..."
    tic = time.time()
    examples = []  # list of (qn_ids, context_ids, ans_span, ans_tokens) triples
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()  # read the next line from each

    context_ids1 = []
    context_tokens1 = []
    qn_ids1 = []
    qn_tokens1= []
    ans_ids1 = []
    ans_tokens1 = []
    anstocontxt_ids1 = []
    anstocontxt_tokens1 = []
    start1_tokens1 = []
    end1_tokens1 = []


    tmp = 0
    while context_line and qn_line and ans_line:  # while you haven't reached the end


        # if tmp == 26:
        #     print tmp
        #
        # tmp += 1
        # Reorganize the map to make the nodes containing the start point comes at the front.
        context_line, start1, end1 = reorganize(context_line, ans_line)
        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context_line, context2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id, is_instr=True)

        ans_tokens, ans_ids = sentence_to_token_ids(ans_line, ans2id)

        anstocontxt_tokens, anstocontxt_ids = sentence_to_token_ids(ans_line, context2id)

        start1_tokens, start1_ids = sentence_to_token_ids(start1, context2id)

        end1_tokens, end1_ids = sentence_to_token_ids(end1, context2id)

        ############# reorganize ans tokens into [start] + [action list] (+ [end]) #####################
        if output_goal:
            ans_tokens = [ans_tokens[0]] + ans_tokens[1::2] + [ans_tokens[-1]]
            ans_ids = [ans_ids[0]] + ans_ids[1::2] + [ans_ids[-1]]
        else:
            ans_tokens = [ans_tokens[0]] + ans_tokens[1::2]
            ans_ids = [ans_ids[0]] + ans_ids[1::2]
        ##############################################################################################s
        

        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        # discard or truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else:  # truncate
                qn_ids = qn_ids[:question_len]

        # discard or truncate too-long contexts
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else:  # truncate
                context_ids = context_ids[:context_len]

        # discard or truncate too-long answer
        if len(ans_ids) > ans_len:
            if discard_long:
                continue
            else:  # truncate
                ans_ids = ans_ids[:ans_len]

        # add to examples
        # examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_ids, ans_tokens, anstocontxt_ids, anstocontxt_tokens))

        context_ids1.append(context_ids)
        context_tokens1.append(context_tokens)
        qn_ids1.append(qn_ids)
        qn_tokens1.append(qn_tokens)
        ans_ids1.append(ans_ids)
        ans_tokens1.append(ans_tokens)
        anstocontxt_ids1.append(anstocontxt_ids)
        anstocontxt_tokens1.append(anstocontxt_tokens)
        start1_tokens1.append(start1_tokens)
        end1_tokens1.append(end1_tokens)

    #     # stop refilling if you have 160 batches
    #     if len(examples) == batch_size * 160:
    #         break
    #
    # # Once you've either got 160 batches or you've reached end of file:
    #
    # # Sort by context length for speed
    # # Note: if you sort by context length, then you'll have batches which contain the same context many times
    # # (because each context appears several times, with different questions)
    # # shuffle==False means to not change the sequence of the input data, thus no sorting.
    # if shuffle:
    #     examples = sorted(examples, key=lambda e: len(e[0]))
    #
    # # Make into batches and append to the list batches
    # for batch_start in xrange(0, len(examples), batch_size):
    #     # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
    #     context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch, anstocontxt_span_batch, anstocontxt_tokens_batch = zip(
    #         *examples[batch_start:batch_start + batch_size])
    #
    #     batches.append(
    #         (context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch, anstocontxt_span_batch, anstocontxt_tokens_batch))
    # if shuffle:
    #     # shuffle the batches
    #     random.shuffle(batches)

    examples = [context_ids1, context_tokens1, qn_ids1, qn_tokens1, ans_ids1, ans_tokens1, anstocontxt_ids1, anstocontxt_tokens1, start1_tokens1, end1_tokens1]
    toc = time.time()
    print "Refilling batches took %.2f seconds" % (toc - tic)
    return examples

# ONLY used for computing vector for triplets in graph.
def compute_graph_embedding(batch_tokens, anstocontxt_tokens, vocab_class, max_len, ans_max_len, start1_tokens, end1_tokens):
    # [flag, node, edge, node]
    dimension = len(vocab_class.flags) + len(vocab_class.nodes) + len(vocab_class.edges) + len(vocab_class.nodes)
    # print "whole dimension {}, node {}, edge {}".format(dimension, len(vocab_class.nodes), len(vocab_class.edges))
    # whole dimension 211, node 78, edge 52
    # if choose to group tokens by edge or node.
    graph_embeddings = np.zeros((len(batch_tokens), max_len, dimension))
    # tgraph_embeddings = np.zeros((len(batch_tokens), max_len, 3))
    context_mask = np.ones((len(batch_tokens), max_len))

    ans_graph_embeddings = np.zeros((len(batch_tokens), ans_max_len, dimension))

    se_graph_embeddings = np.zeros((len(batch_tokens), ans_max_len, 2*dimension))
    # tans_graph_embeddings = np.zeros((len(batch_tokens), ans_max_len, 3))
    ans_mask = np.ones((len(batch_tokens), ans_max_len))

    add_start_end = True

    print("Computing the graph embedding")

    for i, sentence_tokens in enumerate(batch_tokens):
        # print i
        ids = vocab_class.tidy_in_triplet(sentence_tokens)
        sentence_array = np.zeros((max_len, dimension))
        tsentence_array = []
        graph_len = int(len(ids) / 3)

        if graph_len < max_len:
            sentence_array[np.arange(graph_len, max_len), PAD_ID] = 1
            context_mask[i, np.arange(graph_len, max_len)] = 0
        # elif graph_len > max_len:
        #     print "found one long graph.", graph_len
        for number_of_triplets, index in enumerate(range(len(ids))[: max_len * 3: 3]):
            node1, edge, node2 = ids[index: index + 3]
            # for trip in list(itertools.product(*(node1, edge, node2))):
            #     tsentence_array.append(trip)
            for node_id in node1:
                prefix_len = len(vocab_class.flags)
                sentence_array[number_of_triplets, prefix_len + node_id] = 1
            for edge_id in edge:
                prefix_len = len(vocab_class.flags) + len(vocab_class.nodes)
                sentence_array[number_of_triplets, prefix_len + edge_id] = 1
            for node_id in node2:
                prefix_len = len(vocab_class.all_tokens)
                sentence_array[number_of_triplets, prefix_len + node_id] = 1
        # if given the start tokens, we need to append it to the end of graph representation.
        # In case of long graph, we truncate the graph and still append the start tokens.
        # if start_tokens is not None:
        #     last_index = min(number_of_triplets + 1, max_len - 1)
        #     sentence_array[last_index, SOS_ID] = 1
        #     sentence_array[last_index, PAD_ID] = 0
        #     sentence_array[last_index, len(vocab_class.flags) + vocab_class.node2id[start_tokens[i]]] = 1
        #     context_mask[i, last_index] = 1

        if add_start_end:
            sentence_array[0, SOS_ID] = 1
            sentence_array[1, EOS_ID] = 1
            # sentence_array[last_index, len(vocab_class.flags) + vocab_class.node2id[start_tokens[i]]] = 1
            # context_mask[i, last_index] = 1

        graph_embeddings[i, :, :] = sentence_array


    for i, sentence_tokens in enumerate(anstocontxt_tokens):
        ids = vocab_class.ans_tidy_in_triplet(sentence_tokens)
        sentence_array = np.zeros((ans_max_len, dimension))
        graph_len = int(len(ids) / 3)

        if graph_len < ans_max_len:
            sentence_array[np.arange(graph_len, ans_max_len), PAD_ID] = 1
            ans_mask[i, np.arange(graph_len, ans_max_len)] = 0
        # elif graph_len > max_len:
        #     print "found one long graph.", graph_len
        for number_of_triplets, index in enumerate(range(len(ids))[: ans_max_len * 3: 3]):
            node1, edge, node2 = ids[index: index + 3]
            for node_id in node1:
                prefix_len = len(vocab_class.flags)
                sentence_array[number_of_triplets, prefix_len + node_id] = 1
            for edge_id in edge:
                prefix_len = len(vocab_class.flags) + len(vocab_class.nodes)
                sentence_array[number_of_triplets, prefix_len + edge_id] = 1
            for node_id in node2:
                prefix_len = len(vocab_class.all_tokens)
                sentence_array[number_of_triplets, prefix_len + node_id] = 1
        # if given the start tokens, we need to append it to the end of graph representation.
        # In case of long graph, we truncate the graph and still append the start tokens.
        # if start_tokens is not None:
        #     last_index = min(number_of_triplets + 1, max_len - 1)
        #     sentence_array[last_index, SOS_ID] = 1
        #     sentence_array[last_index, PAD_ID] = 0
        #     sentence_array[last_index, len(vocab_class.flags) + vocab_class.node2id[start_tokens[i]]] = 1
        #     context_mask[i, last_index] = 1
        ans_graph_embeddings[i, :, :] = sentence_array

    for i, sentence_tokens in enumerate(start1_tokens):
        ids = vocab_class.tidy_in_triplet(sentence_tokens)
        sentence_array = np.zeros((ans_max_len, 2*dimension))
        graph_len = int(len(ids) / 3)

        # if graph_len < ans_max_len:
        #     sentence_array[np.arange(graph_len, ans_max_len), PAD_ID] = 1
        #     ans_mask[i, np.arange(graph_len, ans_max_len)] = 0
        # elif graph_len > max_len:
        #     print "found one long graph.", graph_len
        for number_of_triplets, index in enumerate(range(len(ids))[: ans_max_len * 3: 3]):
            node1, edge, node2 = ids[index: index + 3]
            for node_id in node1:
                prefix_len = len(vocab_class.flags)
                sentence_array[number_of_triplets, prefix_len + node_id] = 1
            for edge_id in edge:
                prefix_len = len(vocab_class.flags) + len(vocab_class.nodes)
                sentence_array[number_of_triplets, prefix_len + edge_id] = 1
            for node_id in node2:
                prefix_len = len(vocab_class.all_tokens)
                sentence_array[number_of_triplets, prefix_len + node_id] = 1

        ids = vocab_class.tidy_in_triplet(end1_tokens[i])
        # sentence_array = np.zeros((ans_max_len, 2 * dimension))
        graph_len = int(len(ids) / 3)

        # if graph_len < ans_max_len:
        #     sentence_array[np.arange(graph_len, ans_max_len), PAD_ID] = 1
        #     ans_mask[i, np.arange(graph_len, ans_max_len)] = 0
        # elif graph_len > max_len:
        #     print "found one long graph.", graph_len
        for number_of_triplets, index in enumerate(range(len(ids))[: ans_max_len * 3: 3]):
            node1, edge, node2 = ids[index: index + 3]
            for node_id in node1:
                prefix_len = len(vocab_class.flags)
                sentence_array[number_of_triplets, dimension + prefix_len + node_id] = 1
            for edge_id in edge:
                prefix_len = len(vocab_class.flags) + len(vocab_class.nodes)
                sentence_array[number_of_triplets, dimension + prefix_len + edge_id] = 1
            for node_id in node2:
                prefix_len = len(vocab_class.all_tokens)
                sentence_array[number_of_triplets, dimension + prefix_len + node_id] = 1
        #     context_mask[i, last_index] = 1

        for j in range(ans_max_len):
            se_graph_embeddings[i, j, :] = sentence_array[0, :]

    ans_match_ids = np.zeros((len(batch_tokens), ans_max_len, max_len))
    for i in range(graph_embeddings.shape[0]):
        ttp = []
        for j in range(ans_graph_embeddings.shape[1]):
            if ans_mask[i, j] == 1:
                for k in range(graph_embeddings.shape[1]):
                    if context_mask[i, k] == 1:
                        n_ans_graph = np.nonzero(ans_graph_embeddings[i, j, :])
                        ansgraph_i = ans_graph_embeddings[i, j, n_ans_graph]
                        contextgraph_i = graph_embeddings[i, k, n_ans_graph]
                        if np.array_equal(ansgraph_i, contextgraph_i) and not np.isin(k, ttp):
                            ans_match_ids[i, j, k] = 1
                            ttp.append(k)
                            break




    return graph_embeddings, context_mask, ans_graph_embeddings, ans_mask, ans_match_ids, se_graph_embeddings

def get_batch_generator(word2id, context2id, ans2id, context_path, qn_path, ans_path, batch_size, graph_vocab_class,
                        context_len, question_len, answer_len, discard_long, use_raw_graph=True, shuffle=True, output_goal=False):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context2id: dictionary mapping graph symbol to id
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
      use_raw_graph: whether to organize the graph in the unit of triplet.
      shuffle: whether to shuffle the sequence of data.
    """
    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    batches = []

    # shuffle = False

    # # while True:
    # if len(batches) == 0:  # add more batches
    #     refill_batches(word2id, context2id, ans2id, context_file, qn_file, ans_file, batch_size,
    #                    context_len, question_len, answer_len, discard_long, shuffle=shuffle, output_goal=output_goal)
    # if len(batches) == 0:
    #     break

    context_ids, context_tokens, qn_ids, qn_tokens, ans_ids, ans_tokens, anstocontxt_ids, anstocontxt_tokens, start1_tokens, end1_tokens = refill_batches(word2id, context2id, ans2id, context_file, qn_file, ans_file, batch_size,
                   context_len, question_len, answer_len, discard_long, shuffle=shuffle, output_goal=output_goal)
    # Get next batch. These are all lists length batch_size
    # (context_ids, context_tokens, qn_ids, qn_tokens, ans_ids, ans_tokens, anstocontxt_ids, anstocontxt_tokens) = batches

    # Pad context_ids and qn_ids
    qn_ids = padded(qn_ids, question_len)  # pad questions to length question_len
    context_ids = padded(context_ids, context_len)  # pad contexts to length context_len
    ans_ids = padded(ans_ids, answer_len)  # pad ans to maximum length

    # Make qn_ids into a np array and create qn_mask
    qn_ids = np.array(qn_ids)  # shape (batch_size, question_len)
    qn_mask = (qn_ids != PAD_ID).astype(np.int32)  # shape (batch_size, question_len)

    # Make context_ids into a np array and create context_mask
    context_ids = np.array(context_ids)  # shape (batch_size, context_len)
    context_mask = (context_ids != PAD_ID).astype(np.int32)  # shape (batch_size, context_len)

    # Make ans_ids into a np array and create ans_mask
    ans_ids = np.array(ans_ids)
    ans_mask = (ans_ids != PAD_ID).astype(np.int32)

    # Make anstocontxt_ids into a np array and create ans_mask
    anstocontxt_ids = np.array(anstocontxt_ids)
    anstocontxt_mask = (anstocontxt_ids != PAD_ID).astype(np.int32)

    # interpret graph as triplets and append the first token
    if not use_raw_graph:
        context_embeddings, context_mask, ans_graph_embeddings, ans_mask, ans_match_ids, se_graph_embeddings = compute_graph_embedding(
            context_tokens, anstocontxt_tokens, graph_vocab_class, context_mask.shape[1], answer_len, start1_tokens, end1_tokens)

    else:
        context_embeddings = None


    return context_embeddings, ans_match_ids, ans_mask, se_graph_embeddings, qn_ids




class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, word2id, context2id, ans2id, context_path, qn_path, ans_path, batch_size, graph_vocab_class,
                        context_len, question_len, answer_len, discard_long, with_inst=False, use_raw_graph=True, shuffle=True,
                         output_goal=False):
        'Initialization'

        self.context_embeddings, self.ans_match_cat, self.ans_mask, self.se_graph_embeddings, self.qn_ids = get_batch_generator(word2id, context2id, ans2id, context_path, qn_path, ans_path, batch_size, graph_vocab_class,
                        context_len, question_len, answer_len, discard_long, use_raw_graph=use_raw_graph, shuffle=shuffle, output_goal=output_goal)

        # self.context_embeddings = self.datas[0]

        self.batch_size = batch_size

        self.list_IDs = np.arange(self.context_embeddings.shape[0])

        self.shuffle = False

        self.with_inst = with_inst

        # if mod_num>=2:
        #     self.rpnmod = rpnmod

        # if mod_num >=2:
        #     args_dict.bs = 1
        #     self.batch_size = args_dict.bs

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sample_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        if self.with_inst:
            return [self.context_embeddings[sample_ids], self.se_graph_embeddings[sample_ids], self.qn_ids[sample_ids]], \
                   self.ans_match_cat[sample_ids], self.ans_mask[sample_ids]
        else:
            return [self.context_embeddings[sample_ids], self.se_graph_embeddings[sample_ids]], self.ans_match_cat[sample_ids], self.ans_mask[sample_ids]
