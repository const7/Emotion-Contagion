import os
import sys
import pandas as pd
from tqdm import tqdm
from snownlp import sentiment
from clean_data import WeiboPreprocess

neg_corpus_file_path = "../data/snownlp_data/neg.txt"
pos_corpus_file_path = "../data/snownlp_data/pos.txt"
model_save_to = "C:/Users/const/Miniconda3/Lib/site-packages/snownlp/sentiment/sentiment.marshal"

# get corpus from raw weibo data
def make_corpus(rawdata_path="../data/snownlp_data/weibo_senti_100k.csv"):
    # delete files if exist
    if os.path.exists(neg_corpus_file_path):
        os.remove(neg_corpus_file_path)
    if os.path.exists(pos_corpus_file_path):
        os.remove(pos_corpus_file_path)
    # read raw weibo data
    rawdata = pd.read_csv(rawdata_path)
    # data clean
    preprocess = WeiboPreprocess()
    tqdm.pandas()
    rawdata["review"] = rawdata["review"].progress_apply(lambda x: preprocess.clean(x))
    rawdata["len"] = rawdata["review"].progress_apply(lambda x: len(x))
    rawdata = rawdata[rawdata["len"] > 3]
    # write neg/pos review to file
    neg_data = rawdata[rawdata.label == 0]["review"]
    pos_data = rawdata[rawdata.label == 1]["review"]
    neg_data.to_csv(neg_corpus_file_path, index=False, header=False)
    pos_data.to_csv(pos_corpus_file_path, index=False, header=False)

# train snownlp model based on weibo data
def train_weibo_sentiment_model():
    # if os.path.exists(model_save_to):
    #     os.remove(model_save_to)
    sentiment.train(neg_corpus_file_path, pos_corpus_file_path)
    sentiment.save(model_save_to)

if __name__ == "__main__":
    # make_corpus()
    train_weibo_sentiment_model()
