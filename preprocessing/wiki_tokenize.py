import re
import pdb
import sys
from nltk.tokenize import sent_tokenize, word_tokenize


WIKI_EXTRACT_PATH = "/Volumes/ed_seagate_1/projects/wiki-2021/CG/wiki_49"
MIN_ARTICLE_LENGTH = 100
DOC_DELIMITER = re.compile("</doc>\s*<doc.*?>|<doc.*?>|</doc>")


def process_file(path=WIKI_EXTRACT_PATH):
    contents = sys.stdin.read()
    articles = DOC_DELIMITER.split(contents)
    for article in articles:
        if len(article) < MIN_ARTICLE_LENGTH: continue
        for line in article.split("\n"):
            for sentence in sent_tokenize(line.strip()):
                print(" ".join(word_tokenize(sentence)))
        print("")


if __name__ == "__main__":
    process_file()


