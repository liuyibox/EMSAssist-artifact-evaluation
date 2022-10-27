#import Utils as util
import os
import numpy as np
#import yaml
import natsort
from datetime import datetime

def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
    res = []
    for line in lines:
        line = line.strip()
        res.append(line)
    f.close()
    return res

def writeListFile(file_path, output_list, encoding = None):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()

def writeSetFile(file_path, output_set, sort = True):
    output_list = list(output_set)
    if sort:
        output_list.sort()
    writeListFile(file_path, output_list)

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
      ref_f_lines = readFile(ref_file_path)
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
    writeListFile("/home/liuyi/emsAssist_mobisys22/data/text_data/sorted_codes_list.txt", codes_list)
    codes_map = dict()
    for (i, code) in enumerate(codes_list):
      codes_map[code] = i
  
    words_set = set()
    for v in global_d.values():
      words_set.update(v.split())
    words_list = list(words_set)
    words_list = natsort.natsorted(words_list)
    writeListFile("/home/liuyi/emsAssist_mobisys22/data/text_data/sorted_words_list.txt", words_list)
    words_map = dict()
    for (i, word) in enumerate(words_list):
      words_map[word] = i
  
    return d_list, global_d, codes_map, words_map
 
def preprocess_text(text):
    
    '''
        handling punctuations
    '''
    
    text = text.replace("-", " ")
    text = text.replace(",", "")
    text = text.replace(" nos", " not otherwise specified")
    text = text.replace("(", "")
    text = text.replace(")", "")
    if text == "tia":
        text = "transient ischaemic attack"
    return text

def sample_codes_for_all(refresh_training_files = False):


    refinfo = RefInfo()
    if refresh_training_files:
        for f_idx, code_f in enumerate([train_code_f, eval_code_f, test_code_f]):
            lines = []
            with open(code_f) as f:
                for idx, line in enumerate(f):
                    if idx == 0:
                        continue
                    line = line.strip()
                    lines.append("~|~".join(line.split('\t')))
            if f_idx == 0:
                refreshed_file_path = "/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_train.txt"
            elif f_idx == 1:
                refreshed_file_path = "/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_eval.txt"
            elif f_idx == 2:
                refreshed_file_path = "/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_test.txt"    
            writeListFile(refreshed_file_path, lines)
   
    test_lines = readFile("/home/liuyi/emsAssist_mobisys22/data/text_data/no_fitted_separated_desc_code_46_test.txt") 

    # we sample out 1000 * num_spk signs and symptoms from test codes
    spks = ["liuyi", "tian", "amran", "radu", "yichen", "michael"]
    num_spk = len(spks)
    total_sample_num = 1000 * num_spk
    np.random.seed(1993)
    print("total sample num %s" % total_sample_num)
    print("test lines %s" % len(test_lines) )
    sampled_lines = np.random.choice(test_lines, total_sample_num, replace = False)
    assert len(sampled_lines) == len(set(sampled_lines))
    for spk_idx, spk in enumerate(spks):

        print("\n ============== speaker %s ==================" % spk)

        #spk_path = os.path.join("/home/liuyi/tflite_experimental/emsBert/data/text_for_audio_data", spk)
        spk_path = os.path.join("/home/liuyi/emsAssist_mobisys22/data/audio_data/emsAssist_audio_data", spk)
        assert os.path.exists(spk_path)
        spk_lines = sampled_lines[spk_idx*1000 : (spk_idx+1)*1000]
        sampled_100_lines = spk_lines[0:100]
        
        d = refinfo.global_d
        text_lines = []
        text_lines_with_labels = []
        sign_symptom_set = set()
        for line_idx, line in enumerate(sampled_100_lines):
            text_line = []
            event = line.split("~|~")
            assert len(event) == 5
            
            text_line.append(preprocess_text(d[event[0]]))
            sign_symptom_set.add(preprocess_text(d[event[0]]))
            text_line.append(preprocess_text(d[event[1]]))
            sign_symptom_set.add(preprocess_text(d[event[1]]))
        
            for code in event[2].split(' '):
              text_line.append(preprocess_text(d[code]))
              sign_symptom_set.add(preprocess_text(d[code]))
              
            for code in event[3].split(' '):
              text_line.append(preprocess_text(d[code]))
              sign_symptom_set.add(preprocess_text(d[code]))

            text_lines_with_labels.append("\t".join([" ".join(text_line), event[4]]))
            text_lines.append("~|~".join(text_line))

        sampled_100_ss_f_path = os.path.join(spk_path, "sampled_signs_symptoms_100.txt")
        writeListFile(sampled_100_ss_f_path, text_lines)
        print("%s lines to %s" % (len(text_lines), sampled_100_ss_f_path))

        sampled_100_ss_label_f_path = os.path.join(spk_path, "sampled_signs_symptoms_100_with_label.txt")
        writeListFile(sampled_100_ss_label_f_path, text_lines_with_labels)
        print("%s lines to %s" % (len(text_lines_with_labels), sampled_100_ss_label_f_path))

        ss_text_list = list(sign_symptom_set)
        ss_text_list = natsort.natsorted(ss_text_list)
        ss_texts_sample_100_f_path = os.path.join(spk_path, "separete_ps_pi_as_si_texts_for_sample_100.txt") 
        writeListFile(ss_texts_sample_100_f_path, ss_text_list)
        print("%s lines to %s" % (len(ss_text_list), ss_texts_sample_100_f_path))

def convert_transcribed_text_for_emsBert():

    true_text_path = "/home/liuyi/tflite_experimental/emsBert/eval_pretrain/fitted_desc_sampled100e2e_test.tsv"
    transcribed_text_path = "/home/liuyi/TensorFlowASR/examples/conformer/pretrained_librispeech_train_ss_test_concatenated_h5_models_testout.txt"

    true_lines = util.readFile(true_text_path)
    pred_lines = util.readFile(transcribed_text_path)

    assert len(true_lines) == len(pred_lines)
    assert len(true_lines) == 101

    transcribed_e2e_lines = []
    for idx, (true_line, pred_line) in enumerate(zip(true_lines, pred_lines)):
        if idx == 0:
            continue

        transcribed_line = []

        true_text = true_line.split("\t")[0]
        true_label = true_line.split("\t")[1]
        true_text_in_pred = pred_line.split("\t")[2]
        assert true_text == true_text_in_pred

        pred_text = pred_line.split("\t")[3]
        transcribed_line.append(pred_text)
        transcribed_line.append(true_label)
        transcribed_e2e_lines.append(transcribed_line)

    transcribed_e2e_lines.insert(0, ["ps_pi_as_si_desc_c_mml_c", "label"])
    file_path = os.path.join("/home/liuyi/tflite_experimental/emsBert/eval_pretrain", "fitted_desc_sampled100e2e_transcribed_test.tsv")
    util.write2DListFile(file_path, transcribed_e2e_lines, line_sep = "\t")
    print(file_path, len(transcribed_e2e_lines))


if __name__ == "__main__":

    time_s = datetime.now()

    train_code_f = "/home/liuyi/tflite_experimental/emsBert/eval_pretrain/fitted_desc_train_code_template.tsv"
    eval_code_f = "/home/liuyi/tflite_experimental/emsBert/eval_pretrain/fitted_desc_eval_code_template.tsv"
    test_code_f = "/home/liuyi/tflite_experimental/emsBert/eval_pretrain/fitted_desc_test_code_template.tsv"

    sample_codes_for_all(False)
    
#    p.convert_code_to_signs_symptoms()

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)
#    convert_transcribed_text_for_emsBert()

