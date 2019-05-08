"""
Adapted from:
https: // github.com / StanfordVL / behavioral_navigation_nlp

"""

from __future__ import print_function
from collections import Counter, defaultdict
import string
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Interface for summarizing all metrics
# input are two lists of strings of form [start] + [action list] + [goal]
def compute_all_metrics(pred_answer, true_answer):
    # because we only consider the accuracy of the actions, we remove first and last items.
    # prediction_str = normalize_answer(" ".join(pred_answer[1:-1]))
    # ground_truth_str = normalize_answer(" ".join(true_answer[1:-1]))
    prediction_str = normalize_answer(" ".join(pred_answer))
    ground_truth_str = normalize_answer(" ".join(true_answer))
    em = exact_match_score(prediction_str, ground_truth_str)
    f1 = f1_score(prediction_str, ground_truth_str)
    ed = edit_distance(prediction_str.split(), ground_truth_str.split())
    if em > int(pred_answer[-1] == true_answer[-1]):
        print("weird thing happens, pred {}, true {}".format(pred_answer, true_answer))
    return f1, em, ed, pred_answer[-1] == true_answer[-1]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def edit_distance(s1, s2):
    """
    :param s1: list
    :param s2: list
    :return: edit distance of two lists
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def exact_match_score(prediction, ground_truth):
    return prediction == ground_truth

def rough_match_score(prediction, ground_truth):
    prediction = ' '.join(prediction.split(' '))
    ground_truth = ' '.join(ground_truth.split(' '))
    pred_list = prediction.split(' ')
    truth_list = ground_truth.split(' ')
    poss_correct = len(pred_list) == len(truth_list) or \
                   (len(pred_list) > len(truth_list) and pred_list[len(truth_list)] not in ['oor', 'ool'])
    return prediction[: len(ground_truth)] == ground_truth and poss_correct

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
