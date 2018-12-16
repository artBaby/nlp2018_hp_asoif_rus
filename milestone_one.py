from nltk import tokenize
import os
import re
import gensim
from gensim.models.word2vec import LineSentence



BOOKS_DIRECTORY = 'F:/study/Master/2course/NLP/Project/books/asoif/'
RESULT_FILE = 'F:/study/Master/2course/NLP/Project/books/for_Gerhard/hp_result_preproc.txt'
replace_list = ["«", "»", "…", "–", ".\n", ":\n", "..", "—", "* * *", "-", "(", ")"]

all_files = os.listdir("books/asoif/")
print(all_files)

sentences = []
for file in all_files:
    with open(BOOKS_DIRECTORY + file, "r", encoding="utf8") as current_file:
        text = current_file.read()

        for item in replace_list:
            if item == "…" or item == "..":
                text = text.replace(item, ".")
            elif item == "..":
                text = text.replace(item, ".")
            elif item == ".\n":
                text = text.replace(item, ".")
            elif item == ":\n":
                text = text.replace(item, ": ")
            else:
                text = text.replace(item, "")
        # text = text.replace(".\n", " ")
        text = text.replace("\n", " ")
        text = text.replace("  ", " ")

        sentences += tokenize.sent_tokenize(text)

with open(RESULT_FILE, "w", encoding="utf8") as result_file:
    for sent in sentences:
        result_file.write(re.sub('^\.', '', sent) + "\n")

print("Training . . . ")
sentences = LineSentence(RESULT_FILE)
# model = gensim.models.Word2Vec(sentences, min_count=5, size=300, workers=4, window=10, sg=1, negative=5)
model = gensim.models.Word2Vec(sentences)
print(model.corpus_count)
print(len(model.wv.vocab))

model.wv.save_word2vec_format("ResultHP_default.model", binary=True)

