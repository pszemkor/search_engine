import numpy as  np
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from itertools import groupby


def divide_to_files():
    with open("corpus.txt", encoding="utf8") as f:
        content = f.readlines()
        # print(content)
    lists = [list(group) for k, group in groupby(content, lambda x: x == "\n") if not k]
    # print(lists)
    for file in lists:
        path = "data/"+file[0].strip().replace('/', '_').replace('\\', '_')+ '.txt'
        handle = open(path, "w+", encoding="utf8")
        for line in file:
            handle.write(line)
        handle.close()


divide_to_files()
