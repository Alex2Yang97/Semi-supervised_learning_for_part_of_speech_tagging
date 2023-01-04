# /**
#  * @author Zhirui(Alex) Yang
#  * @email zy2494@columbia.edu
#  * @create date 2022-12-08 13:07:04
#  * @modify date 2022-12-08 13:07:04
#  * @desc [description]
#  */


import os
import codecs


# Raw data folders
PJ_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(PJ_DIR, "data", "gweb_sancl")

POS_FINE_DIR = os.path.join(DATA_DIR, "pos_fine")
UNLABELED_DIR = os.path.join(DATA_DIR, "unlabeled")

PF_WSJ_DIR = os.path.join(POS_FINE_DIR, "wsj")

# User's folders
MODEL_DIR = os.path.join(PJ_DIR, "model")
INT_RESULT_DIR = os.path.join(PJ_DIR, "intermediate_result")
RESULT_DIR = os.path.join(PJ_DIR, "result")
METRICS_DIR = os.path.join(PJ_DIR, "metrics")
PLOT_TAGS_DIR = os.path.join(PJ_DIR, "plots_tags")


DOMAIN_LST = ["answers", "emails", "newsgroups", "reviews", "weblogs"]

def read_conll_file(file_name, raw=False):
    """
    read in conll file
    word1    tag1
    ...      ...
    wordN    tagN
    Sentences MUST be separated by newlines!
    :param file_name: file to read in
    :param raw: if raw text file (with one sentence per line) -- adds 'DUMMY' label
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    current_words = []
    current_tags = []
    
    for line in codecs.open(file_name, encoding='utf-8'):
        #line = line.strip()
        line = line[:-1]

        if line:
            if raw:
                current_words = line.split() ## simple splitting by space
                current_tags = ['DUMMY' for _ in current_words]
                yield (current_words, current_tags)

            else:
                if len(line.split("\t")) != 2:
                    if len(line.split("\t")) == 1: # emtpy words in gimpel
                        raise IOError("Issue with input file - doesn't have a tag or token?")
                    else:
                        print("erroneous line: {} (line number: {}) ".format(line), file=sys.stderr)
                        exit()
                else:
                    word, tag = line.split('\t')
                current_words.append(word)
                current_tags.append(tag)

        else:
            if current_words and not raw: #skip emtpy lines
                yield (current_words, current_tags)
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        yield (current_words, current_tags)


def read_data(data_file):
    word_lst = []
    tag_lst = []
    tags = []
    for word, tag in read_conll_file(data_file):
        word_lst.append(word)
        tag_lst.append(tag)
        tags.extend(tag)
    print("The number of samples:", len(word_lst))
    print("The number of tags", len(set(tags)))
    return word_lst, tag_lst, list(set(tags))


def read_unlabeled_data(file_path, max_unlabeled=False):
  data = []
  with open(file_path, 'rb') as f:
    for line in f:
      if max_unlabeled and len(data) == max_unlabeled:
        break
      line = line.decode('utf-8','ignore').strip().split()
      data.append(line)
  print('Loaded... {} unlabeled instances'.format(len(data)))
  return data
  

# Read wsj data
wsj_train_file = os.path.join(PF_WSJ_DIR, "gweb-wsj-train.conll")
wsj_dev_file = os.path.join(PF_WSJ_DIR, "gweb-wsj-dev.conll")
wsj_test_file = os.path.join(PF_WSJ_DIR, "gweb-wsj-test.conll")
wsj_train_word_lst, wsj_train_tag_lst, wsj_train_tag_set = read_data(wsj_train_file)
wsj_dev_word_lst, wsj_dev_tag_lst, wsj_dev_tag_set = read_data(wsj_dev_file)
wsj_test_word_lst, wsj_test_tag_lst, wsj_test_tag_set = read_data(wsj_test_file)


wsj_tags = wsj_train_tag_set + wsj_dev_tag_set + wsj_test_tag_set
wsj_tags = sorted(list(set(wsj_tags)))
WSJ_TAGS = ["<pad>"] + wsj_tags

TAG2IDX = {tag:idx for idx, tag in enumerate(WSJ_TAGS)}
IDX2TAG = {idx:tag for idx, tag in enumerate(WSJ_TAGS)}


def filter_tag(process_words, process_tags, label_tags_set=WSJ_TAGS):
  new_words = []
  new_tags = []
  for words, tags in zip(process_words, process_tags):
    w_lst = []
    t_lst = []
    for i, t in enumerate(tags):
      if t in label_tags_set:
        w_lst.append(words[i])
        t_lst.append(tags[i])

    if w_lst:
      new_words.append(w_lst)
      new_tags.append(t_lst)
  print("after filter tag", len(new_words))
  return new_words, new_tags


def create_sub_dir(domain, method_name="Online_fixed_self_learning"):
    # create dir
    sub_model_dir = os.path.join(MODEL_DIR, method_name, domain)
    if not os.path.isdir(sub_model_dir):
        os.makedirs(sub_model_dir)
        print("Create", sub_model_dir)

    sub_metrics_dir = os.path.join(METRICS_DIR, method_name, domain)
    if not os.path.isdir(sub_metrics_dir):
        os.makedirs(sub_metrics_dir)
        print("Create", sub_metrics_dir)

    sub_result_dir = os.path.join(RESULT_DIR, method_name, domain)
    if not os.path.isdir(sub_result_dir):
        os.makedirs(sub_result_dir)
        print("Create", sub_result_dir)

    sub_plots_tags_dir = os.path.join(PLOT_TAGS_DIR, method_name, domain)
    if not os.path.isdir(sub_plots_tags_dir):
        os.makedirs(sub_plots_tags_dir)
        print("Create", sub_plots_tags_dir)

    sub_int_res_dir = os.path.join(INT_RESULT_DIR, method_name, domain)
    if not os.path.isdir(sub_int_res_dir):
        os.makedirs(sub_int_res_dir)
        print("Create", sub_int_res_dir)
    
    return sub_model_dir, sub_metrics_dir, sub_result_dir, sub_plots_tags_dir, sub_int_res_dir