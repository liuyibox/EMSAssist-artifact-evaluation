import numpy as np
import os
import collections
import csv
from absl import logging
logging.set_verbosity(logging.INFO)
import tensorflow as tf
assert tf.__version__.startswith('2')
import tensorflow_addons as tfa

import file_util
import tokenization
import optimization
import configs as bert_configs
import quantization_configs as quant_configs
#import metadata_writer_for_bert_text_classifier as bert_metadata_writer
import tensorflow_hub as hub
from datetime import datetime
import random
import tempfile
import argparse
import natsort
import time

#import TFUtils as tfutil
import pandas as pd
#import Utils as util
from ranking_func import rank
from scipy import spatial
import Utils as util
import math

global_seed = 1993


speakers = ["liuyi", "tian", "yichen"]

#def readFile(file_path, encoding = None):
#    f = open(file_path, 'r')
#    lines = f.read().splitlines()
#    res = []
#    for line in lines:
#        line = line.strip()
#        res.append(line)
#    f.close()
#    return res

def writeListFile(file_path, output_list):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

def computeSimilarity(tv, pvs, pids, metric = "cosine"):

    ranking_list = []
    for i in range(pvs.shape[0]):
        pv = pvs[i]
        cur_id = pids[i]
        sim = 0.0
        if metric == "cosine":
            sim = 1 - spatial.distance.cosine(tv, pv)
            if math.isnan(sim):
                sim = 0.0
        elif metric == "dot":
            sim = np.dot(tv, pv)
        else:
            print("we require a metric: cosine, dot_product")
            exit(1)
            
        ranking_list.append((cur_id, sim))
        
    return ranking_list

def get_top_score(pvs, tvs, pids, tids, args):
    top1 = 0.0
    top3 = 0.0
    top5 = 0.0   
    total_count = tvs.shape[0]
    for idx in range(tvs.shape[0]):

        tv = tvs[idx]
        tp = tids[idx]
        ranking = computeSimilarity(tv, pvs, pids, metric = args.metric)
        # rank
        candi_list, score_list = rank(ranking)[0], rank(ranking)[1]
#        if idx == 0:
#            print(candi_list, score_list)

        top1set = set()
        top3set = set()
        top5set = set()
        # for each input, we get an top-1, top-3, top-5 score, respectively
        num = sum(score_list[:5])
        for candi_idx, candi_id in enumerate(candi_list):
#            candi_id_st = set(candi_id.split(';'))
            if candi_idx >= 5:
                break
            candi_score = (0.0 if num == 0.0 else score_list[candi_idx] / num)
            if candi_idx == 0:
                top1set.add(candi_id)
            if candi_idx <= 2:
                top3set.add(candi_id)
            top5set.add(candi_id)
#            out_count += 1

        if tp in top1set != 0:
            top1 += 1.0

        if tp in top3set != 0:
            top3 += 1.0

        if tp in top5set != 0:
            top5 += 1.0

    return top1/total_count, top3/total_count, top5/total_count
        

def getFittedInfo():

    #protocol_csv_file = os.path.join(home_dir, 'Fit_NEMSIS_To_TAMU_Revision1.tsv')
    protocol_csv_file = '/home/liuyi/MetamapMatching/Fit_NEMSIS_To_TAMU_Revision1.tsv'

    # original tamu protocol set size if 108
    tamu_text_f = '/home/liuyi/MetamapMatching/revision1_text.txt'
    tamu_texts = util.readFile(tamu_text_f)

    tamu_text_c_f = '/home/liuyi/MetamapMatching/revision1_text_metamap_concepts.txt'
    tamu_text_c = util.readNEMSISFile(tamu_text_c_f, discard_header = False)

    tamu_text_mmlc_f = '/home/liuyi/MetamapMatching/revision1_text_metamaplite_concepts.txt'
    tamu_text_mmlc = util.readNEMSISFile(tamu_text_mmlc_f, discard_header = False)

    nemsis_id2text = dict()
    nemsis_id2c = dict()
    nemsis_id2mmlc = dict()
    
    protocol_list = pd.read_csv(protocol_csv_file, sep = '\t', dtype = str)
    protocol_list = protocol_list[['TAMU Protocol ID', 'TAMU Protocol', 'NEMSIS Protocol ID', 'NEMSIS Protocol', 'Signs&Symptoms', 'History']]
    protocol_list = protocol_list.dropna()

    idx = 0
    nemsis_id_set = set()
    for _, row in protocol_list.iterrows():

        cur_tamu_id = row['TAMU Protocol ID']
        cur_tamu_name = row['TAMU Protocol']
        cur_nemsis_id = row['NEMSIS Protocol ID']
        cur_nemsis_name = row['NEMSIS Protocol']
        cur_sign = row['Signs&Symptoms']
#        cur_hist = row['History']

#        if cur_tamu_id == "" or cur_tamu_name == "" or cur_nemsis_id == "" or cur_nemsis_name == "" or cur_sign == "" or cur_hist == "":
        if cur_tamu_id == "" or cur_tamu_name == "" or cur_nemsis_id == "" or cur_nemsis_name == "" or cur_sign == "":
            continue

        nemsis_id_set.add(cur_nemsis_id)

        cur_text = tamu_texts[idx]          # str
        cur_c = tamu_text_c[idx]            # list
        cur_mmlc = tamu_text_mmlc[idx]      # list

        if cur_nemsis_id in nemsis_id2c:     # id - concept_list
            assert(isinstance(nemsis_id2c[cur_nemsis_id], list))
            nemsis_id2c[cur_nemsis_id].extend(cur_c)
        else:
            nemsis_id2c[cur_nemsis_id] = cur_c

        if cur_nemsis_id in nemsis_id2mmlc:     # id - concept_list
            assert(isinstance(nemsis_id2mmlc[cur_nemsis_id], list))
            nemsis_id2mmlc[cur_nemsis_id].extend(cur_mmlc)
        else:
            nemsis_id2mmlc[cur_nemsis_id] = cur_mmlc

        if cur_nemsis_id in nemsis_id2text:
            assert(isinstance(nemsis_id2text[cur_nemsis_id], str))
            v = nemsis_id2text[cur_nemsis_id]
            nemsis_id2text[cur_nemsis_id] = v + ' ' + cur_text
        else:
            nemsis_id2text[cur_nemsis_id] = cur_text

        idx += 1


    nemsis_id_list = list(nemsis_id_set)
    nemsis_id_list.sort()

    fitted_c_list = []
    fitted_mmlc_list = []
    fitted_text_list = []
    for nemsis_id in nemsis_id_list:
        c = nemsis_id2c[nemsis_id]          # c is a list
        fitted_c_list.append(c)
        mmlc = nemsis_id2mmlc[nemsis_id]    # mmlc is a list
        fitted_mmlc_list.append(mmlc)
        text = nemsis_id2text[nemsis_id]    # text is a string
        fitted_text_list.append(text)

    return fitted_c_list, fitted_mmlc_list, fitted_text_list, nemsis_id_list


class RefInfo(object):

  def __init__(self):

    nemsis_dir = "/slot1/NEMSIS-files/"
    pri_sym_ref = os.path.join(nemsis_dir, "ESITUATION_09REF.txt")
    pri_imp_ref = os.path.join(nemsis_dir, "ESITUATION_11REF.txt")
    add_sym_ref = os.path.join(nemsis_dir, "ESITUATION_10REF.txt")
    sec_imp_ref = os.path.join(nemsis_dir, "ESITUATION_12REF.txt")

    self.ref_files = [pri_sym_ref, pri_imp_ref, add_sym_ref, sec_imp_ref]
    self.d_list, self.global_d, self.code_map, self.word_map = self.get_dict() 

  def get_dict(self):

    d_list = []
    global_d = dict()

    for ref_idx, ref_file_path in enumerate(self.ref_files):
      ref_f_lines = util.readFile(ref_file_path)
      ref_f_lines = [s.split("~|~") for s in ref_f_lines]
      d = dict()

      for i, line in enumerate(ref_f_lines):
        if i == 0:
          continue
        k = line[0].strip()
        v = line[1].strip()
        v = v.lower()

        if k in d:
          assert d[k] == v
        d[k] = v

        if k in global_d:
          assert global_d[k] == v
        global_d[k] = v

      d_list.append(d)

    codes_list = list(global_d.keys())
    codes_list = natsort.natsorted(codes_list)
    codes_map = dict()
    for (i, code) in enumerate(codes_list):
      codes_map[code] = i
  
    words_set = set()
    for v in global_d.values():
      words_set.update(v.split())
    words_list = list(words_set)
    words_list = natsort.natsorted(words_list)
    words_map = dict()
    for (i, word) in enumerate(words_list):
      words_map[word] = i
  
    return d_list, global_d, codes_map, words_map
 
def build_vocab_tokenizer(model_uri, do_lower_case):
  """Builds the class. Used for lazy initialization."""
  vocab_file = os.path.join(model_uri, 'assets', 'vocab.txt')
  assert(tf.io.gfile.exists(vocab_file))

  tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
  return vocab_file, tokenizer

def codes_to_texts(codes_lines, refinfo):

  d = refinfo.global_d
  text_lines = []
  word_lines = []
  code_lines = []

  for idx, codes_line in enumerate(codes_lines):
    text_line = []
    word_line = []
    code_line = []

    event = codes_line.split("~|~")
    assert len(event) == 5

    text_line.append(d[event[0]])
    word_line.extend(d[event[0]].split())
    code_line.append(event[0])

    text_line.append(d[event[1]])
    word_line.extend(d[event[1]].split())
    code_line.append(event[1])

    for code in event[2].split(' '):
      text_line.append(d[code])
      word_line.extend(d[code].split())
    code_line.extend(event[2].split(' '))
      
    for code in event[3].split(' '):
      text_line.append(d[code])
      word_line.extend(d[code].split())
    code_line.extend(event[3].split(' '))

    text_lines.append([" ".join(text_line), event[4]])
    word_lines.append([word_line, event[4]])
    code_lines.append([code_line, event[4]])

  return text_lines, word_lines, code_lines

def get_single_feature(args, text, label, label_map):
    
    label_id = label_map[label]
    _, tokenizer = build_vocab_tokenizer(args.init_model, do_lower_case = True)
    tokens_a = tokenizer.tokenize(text)
    if len(tokens_a) > args.max_seq_len - 2:
      tokens_a = tokens_a[0:(args.max_seq_len - 2)]    

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)    

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < args.max_seq_len:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)

def get_features(args, lines: list, label_list):
    
#    meta_data = file_util.load_json_file(args.meta_data_file)

    feature_list = []
    for idx, line in enumerate(lines):
        text, label = line.split("~|~")[0], line.split("~|~")[1]
        feature = get_single_feature(text, label, label_map)
        feature_list.append(feature)

    return feature_list
        
def evaluate_concept_matching(args):
    
    fitted_c_list, fitted_mmlc_list, fitted_text_list, nemsis_id_list = getFittedInfo()
    print("fitted_c_list %s, fitted_mmlc_list %s, fitted_text_list %s, nemsis_id_list %s" % (len(fitted_c_list), len(fitted_mmlc_list), len(fitted_text_list), len(nemsis_id_list)))
#    print("nemsis_id_list: %s" % nemsis_id_list)

    _, tokenizer = build_vocab_tokenizer(args.init_model, do_lower_case = True)
    label_map = {}
    for i, label in enumerate(nemsis_id_list):
        label_map[label] = i

    lines = util.readFile(args.test_file)
    ivs = np.zeros((len(lines), len(tokenizer.vocab)), dtype=np.int32)
    input_labels = []
    input_label_ids = np.zeros(len(lines), dtype=np.int32)
    input_texts = []
    for idx, line in enumerate(lines):
        text, label = line.split("~|~")[0], line.split("~|~")[1]    
        tokens = tokenizer.tokenize(text)
#        if idx == 0:
#            print("idx = %s, tokens(%s): %s" % (idx, len(tokens), tokens))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        for token_id in token_ids:
            ivs[idx, token_id] += 1

        non_zero_cols = []
#        if idx == 0:
#            for col, val in enumerate(ivs[idx]):
#                if val != 0:
#                    non_zero_cols.append(col)
#            print("idx = %s, col indices(%s): %s" % (idx, len(non_zero_cols), non_zero_cols))
        input_labels.append(label)
        label_id = label_map[label]
        input_label_ids[idx] = label_id
    print(input_label_ids)
    print("ivs shape ", ivs.shape)

    pvs = np.zeros((len(nemsis_id_list), len(tokenizer.vocab)), dtype=np.int32)
    for idx, text in enumerate(fitted_text_list):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        for token_id in token_ids:
            pvs[idx, token_id] += 1
    print("pvs shape ", pvs.shape)

#    t_top1, t_top3, t_top5 = get_top_score(pvs, ivs, nemsis_id_list, input_label_ids, args)
    t_top1, t_top3, t_top5 = get_top_score(pvs, ivs, nemsis_id_list, input_labels, args)
    print("input token topk: ", t_top1, t_top3, t_top5)

if __name__ == "__main__":
    
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for EMSBert")

    parser.add_argument("--init_model", action='store', type=str, default = "/home/liuyi/tflite_experimental/train_pipeline_standalone/init_models/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT_1", help="directory storing the different initialization models")
    parser.add_argument("--test_file", action='store', type=str, help = "indicate which file to test with models", required = True)
    parser.add_argument("--metric", action='store', type=str, default = 'cosine', help = "indicate which similarity method will be used")

    args = parser.parse_args()

    evaluate_concept_matching(args)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

   
