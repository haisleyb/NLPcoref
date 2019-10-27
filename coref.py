import nltk
from nltk.corpus import wordnet
from nltk import tree
from nltk.parse import RecursiveDescentParser
import sys


class coref_file:
    def __init__(self, file_name, contents):
        self.name = file_name
        self.sentences = contents
        self.corefs = []
        self.coref_dict = {}
        self.regular_words = []
        self.START_COREF_TAG = "<COREF ID="
        self.END_COREF_TAG = "</COREF>"

    def find_corefs(self):
        i = 0
        for sentence in self.sentences:
            start_index = -1
            sentence_words = []
            while True:
                start_index = sentence.find(self.START_COREF_TAG)
                if start_index < 0:
                    if len(sentence) > 0:
                        sentence_words.append(sentence)
                    break
                sentence_words.append(sentence[:start_index])
                temp = sentence.find(">")
                end_index = sentence.find(self.END_COREF_TAG)
                head_coref = sentence[start_index + (temp - start_index + 1):end_index]
                self.corefs.append((head_coref, i))
                sentence = sentence[end_index + 8:]
                i += 1

            self.regular_words.append(sentence_words)

        return self.corefs

    def string_match(self):
        for noun_t in self.corefs:
            noun = noun_t[0]
            for sentence in self.regular_words:
                for string in sentence:
                    if noun in string:
                        index = string.find(noun)
                        matched_word = string[index:index+len(noun)]
                        # Tuple of word and index of sentence it came from
                        t = (matched_word, self.regular_words.index(sentence))
                        if noun in self.coref_dict.keys():
                            self.coref_dict[noun].append(t)
                        else:
                            self.coref_dict[noun] = [t]

    def print_result(self):
        for coref in self.corefs:
            noun = coref[0]
            index = coref[1]
            print('<COREF ID="X%d">%s</COREF>' % (index, noun))
            if noun not in self.coref_dict.keys():
                print()
                continue
            lists = self.coref_dict[noun]
            for word in lists:
                w_index = word[1]
                w_noun = word[0]
                print('{%d} {%s}' % (w_index, w_noun))
            print()




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
        c_file = coref_file(file_name, sentences)

        c_file.find_corefs()

        c_file.string_match()

        c_file.print_result()


main()