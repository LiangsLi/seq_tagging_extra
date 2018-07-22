import sys
import argparse
from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
import pickle

import logging
logger = logging.getLogger(__name__)


def train_fasttext(input_fp, output_fp, args):
    fasttext_fp = args.fasttext_fp
    paras = dict(model='skipgram', size=50, alpha=0.1,
                 window=3, min_count=2, loss='ns', sample=1e-4, iter=400, min_n=2, max_n=3, threads=4)
    model = FastText.train(fasttext_fp, input_fp, output_fp, **paras)
    model.save(output_fp)


def train_word2vec(input_fp, output_fp, args):
    sentences = pickle.load(open(input_fp, 'rb'))
    # sentences = [line.strip().split() for line in open(input_fp)]
    paras = dict(size=100, window=7, min_count=2, sg=1, iter=400, sample=1e-4, alpha=0.025)
    model = Word2Vec(sentences, **paras)
    model.save(output_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=False, default='word2vec', help='模型名称')
    parser.add_argument('-i', '--input', required=True, help='训练语料，每行一个句子（每个词以空格分割）')
    parser.add_argument('-o', '--output', required=True, help='词向量模型的输出路径（注：fasttext下文件名不要带后缀）')
    parser.add_argument('-f', '--fasttext_fp', required=False, help='fasttext C++代码所在位置',
                        default='/Users/king/Documents/Useful_Softwares/fastText/fasttext')
    args = parser.parse_args()

    input, output = args.input, args.output

    train_op = getattr(sys.modules[__name__], 'train_' + args.model)
    train_op(input, output, args)
