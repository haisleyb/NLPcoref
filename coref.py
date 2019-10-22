import nltk
from nltk.corpus import names
from nltk import tree
import sys


class coref:
    def __init__(self, file_name, contents):
        self.name = file_name
        self.sentences = contents


'''Takes in the name of a file, reads it, and returns a list of each line.'''
def parse_file_lines(name):
    f = open(name, "r")
    lines = f.readlines()
    files = []
    f.close()
    for line in lines:
        files.append(line.strip('\n'))
    return files

'''Removes the S tags from each sentence. IDs are kept track by the index position in the list.'''
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

    # Get a list of all files we're reading
    input_files = parse_file_lines(list_file)
    corefs = []
    # Create a coref object for each file
    for file_name in input_files:
        # Get all sentences
        contents = parse_file_lines(file_name)
        # Clean up the S tags
        sentences = remove_s_tag(contents)
        # Create the coref object
        c = coref(file_name, sentences)
        # Add to our list for further processing
        corefs.append(c)


main()