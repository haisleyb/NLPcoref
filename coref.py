import nltk
from nltk.corpus import names
from nltk import tree
import sys


class coref:
    def __init__(self, file_name, contents):
        self.name = file_name
        self.sentences = contents


def parse_file_lines(name):
    f = open(name, "r")
    lines = f.readlines()
    files = []
    f.close()
    for line in lines:
        files.append(line.strip('\n'))
    return files

def remove_s_tag(lines):
    sentences = []
    i = 0
    for line in lines:
        line = line.replace('<S ID="' + str(i) + '">', '')
        s = line.replace('</S>', '')
        sentences.append(s)
        i += 1
    return sentences


def main():
    '''
    if len(sys.argv) >= 3:
        list_file = sys.argv[1]
        response_dir = sys.argv[2]
    else:
        print("Missing arguments")
        sys.exit(-1)
    '''

    list_file = "test.listfile"
    response_dir = "responses"
    input_files = parse_file_lines(list_file)
    corefs = []
    for file_name in input_files:
        contents = parse_file_lines(file_name)
        sentences = remove_s_tag(contents)
        c = coref(file_name, sentences)
        corefs.append(c)



main()