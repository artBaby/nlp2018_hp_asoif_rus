import sys

import gensim
from gensim.models import FastText
from gensim.models.word2vec import LineSentence

SAVE_PATH = "./models/"
BOOKS_PATH = "./books/"
EXTENSION = ".model"

def create_models(book_name):
    print("Creating models for " + book_name.upper() + " . . . ")
    sentences = LineSentence(BOOKS_PATH + book_name + "_result.txt")

    print("Training default w2v . . . ")
    model = gensim.models.Word2Vec(sentences)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_default_w2v" + EXTENSION, binary=True)

    print("Training w2v 1 . . . ")
    model = gensim.models.Word2Vec(sentences, size=300, negative=0, sg=1, hs=1, iter=15)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_w2v_1" + EXTENSION, binary=True)

    print("Training w2v 2 . . . ")
    model = gensim.models.Word2Vec(sentences, size=300, negative=0, sg=1, hs=1, iter=15, window=12)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_w2v_2" + EXTENSION, binary=True)

    print("Training w2v 3 . . . ")
    model = gensim.models.Word2Vec(sentences, size=300, negative=15, sg=1, hs=1, iter=15, window=12)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_w2v_3" + EXTENSION, binary=True)

    print("Training w2v 4 . . . ")
    model = gensim.models.Word2Vec(sentences, size=300, negative=0, sg=0, hs=1, iter=15)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_w2v_4" + EXTENSION, binary=True)

    print("Training ft default . . . ")
    model = FastText(sentences)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_default_ft" + EXTENSION, binary=True)

    print("Training ft 1 . . . ")
    model = FastText(sentences, sg=1, hs=1, size=300, iter=15, window=12, negative=0)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_ft_1" + EXTENSION, binary=True)

    print("Training ft 2 . . . ")
    model = FastText(sentences, sg=1, hs=1, size=300, iter=15, window=12, negative=15)
    model.wv.save_word2vec_format(SAVE_PATH + book_name + "_ft_2" + EXTENSION, binary=True)
    print("Training successfully\n")
