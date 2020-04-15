import os
import sys
import pandas as pd
from tqdm import tqdm
from snownlp import sentiment, SnowNLP
from clean_data import WeiboPreprocess

weibo_senti_corpus_path = "../data/snownlp_data/weibo_senti_100k.csv"
neg_corpus_file_path = "../data/snownlp_data/neg.txt"
pos_corpus_file_path = "../data/snownlp_data/pos.txt"
eval_corpus_path = "../data/snownlp_data/eval_data.csv"
model_save_to = "C:/Users/const/Miniconda3/Lib/site-packages/snownlp/sentiment/sentiment.marshal"

# get corpus from raw weibo data
def make_corpus(rawdata_path, neg_corpus_file, pos_corpus_file):
    # delete files if exist
    if os.path.exists(neg_corpus_file):
        os.remove(neg_corpus_file)
    if os.path.exists(pos_corpus_file):
        os.remove(pos_corpus_file)
    if os.path.exists(eval_corpus_path):
        os.remove(eval_corpus_path)
    # read raw weibo data
    data = pd.read_csv(rawdata_path)
    # clean data
    tqdm.pandas()
    preprocess = WeiboPreprocess()
    print("------------------------------------ process data ------------------------------------")
    data["review"] = data["review"].progress_apply(lambda x: preprocess.clean(x))
    data = data[data["review"].str.len() > 3]
    # select train data and test data with 8:2
    train_data = data.sample(frac=0.8, random_state=0, axis=0)
    test_data = data[~data.index.isin(train_data.index)]
    # wirte train data to file
    neg_data = train_data[train_data.label == 0]["review"]
    pos_data = train_data[train_data.label == 1]["review"]
    neg_data.to_csv(neg_corpus_file, index=False, header=False)
    pos_data.to_csv(pos_corpus_file, index=False, header=False)
    # write test data to file
    test_data.to_csv(eval_corpus_path, index=False)

# train snownlp model based on weibo data
def train_weibo_sentiment_model(neg_corpus_file, pos_corpus_file, model_path):
    print("------------------------------------ train ------------------------------------")
    sentiment.train(neg_corpus_file, pos_corpus_file)
    sentiment.save(model_path)

def eval_model(eval_corpus):
    eval_data = pd.read_csv(eval_corpus)
    tqdm.pandas()
    print("------------------------------------ eval ------------------------------------")
    eval_data["predict"] = eval_data["review"].progress_apply(lambda x: 1 if SnowNLP(x).sentiments >= 0.5 else 0)
    eval_data["correct"] = eval_data[["label", "predict"]].progress_apply(lambda x: 1 if x["label"] == x["predict"] else 0, axis=1)
    print("准确率：{}".format(eval_data["correct"].sum() / eval_data.shape[0]))

if __name__ == "__main__":
    # make_corpus(weibo_senti_corpus_path, neg_corpus_file_path, pos_corpus_file_path)
    # train_weibo_sentiment_model(neg_corpus_file_path, pos_corpus_file_path, model_save_to)
    eval_model(eval_corpus_path)