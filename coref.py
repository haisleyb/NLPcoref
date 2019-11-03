import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, ne_chunk
import re
import warnings
from nltk.corpus import names
from nltk import Tree
import string

from nltk.corpus import stopwords
from fuzzywuzzymit import fuzz
from fuzzywuzzymit import process

from nltk.stem import WordNetLemmatizer
from nltk.parse.corenlp import CoreNLPParser
import sys


class coref_file:
    def __init__(self, file_name, contents, response_dir):
        self.name = file_name.split('.')[0].split('/')[-1]
        # Original sentences with the xml
        self.sentences = contents
        # Sentences without xml
        self.cleaned_sentences = []
        # Tagged sentences
        self.tagged_sentences = []
        # Dictionary with coref noun as the key and number as a value
        self.coref_index = {}
        # Dictionary with sentence number as a key and a list of corefs in them
        self.coref_sentence = {}
        # Dictionary of corefs and what they've resolved to
        self.coref_resolved = {}
        # Stop words
        self.stop_words = set(stopwords.words('english'))
        self.START_COREF_TAG = "<COREF ID="
        self.END_COREF_TAG = "</COREF>"
        self.response = response_dir


    def build_tag(self, head, i):
        return self.START_COREF_TAG + '"X' + str(i) + '">' + head + self.END_COREF_TAG

    def tag_sentence(self):
        for sentence in self.cleaned_sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            self.tagged_sentences.append(tagged)
            #chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            #chunkParser = nltk.RegexpParser(chunkGram)
            #chunked = chunkParser.parse(tagged)

    def find_synonyms(self, words):
        # Build our nouns list of synonyms
        synonyms = []
        for f in words:
            for syn in wordnet.synsets(f):
                for l in syn.lemmas():
                    if l.name() not in synonyms:
                        synonyms.append(l.name().replace("_", " "))
        return synonyms

    def get_continuous_chunks(self, text, label):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        prev = None
        continuous_chunk = []
        current_chunk = []
        for subtree in chunked:
            if type(subtree) == Tree and subtree.label() == label:
                current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def find_corefs(self):
        i = 0
        sent = 0
        for sentence in self.sentences:
            self.coref_sentence[sent] = []
            start_index = -1
            while True:
                start_index = sentence.find(self.START_COREF_TAG)
                if start_index < 0:
                    # Removes punctuation
                    #sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                    self.cleaned_sentences.append(sentence)
                    break
                temp = sentence.find(">")
                end_index = sentence.find(self.END_COREF_TAG)
                head_coref = sentence[start_index + (temp - start_index + 1):end_index]
                tag = self.build_tag(head_coref, i)
                sentence = sentence.replace(tag, "X" + str(i))
                self.coref_index[head_coref] = i
                self.coref_sentence[sent].append(head_coref)
                self.coref_resolved[head_coref] = []
                i += 1
            sent += 1
        return self.coref_index

    def string_match(self):
        for loc in self.coref_sentence.keys():
            # Get the list of corefs in the current sentence
            heads = self.coref_sentence[loc]
            for noun in heads:
                # Get the coref number
                num = self.coref_index[noun]
                # Tokenized for partial matching
                words = word_tokenize(noun)
                # Remove stop words
                filtered = []
                if len(words) > 1:
                    for w in words:
                        if w not in self.stop_words:
                            filtered.append(w)
                else:
                    filtered.append(noun)
                # Only loop through the sentence the coref is in and onwards
                i = loc
                for sentence in self.cleaned_sentences[loc:]:
                    # Check if coref is in the sentence
                    index = sentence.find("X" + str(num))
                    if index >= 0:
                        # Find where the coref begins in the sentence
                        sentence = sentence[index + 3:]
                    # Search exact strings
                    if noun.lower() in sentence.lower():
                        index = sentence.lower().find(noun.lower())
                        matched_word = sentence[index:index + len(noun)]
                        # Tuple of word and sentence number it came from
                        t = (matched_word, i)
                        if t not in self.coref_resolved[noun]:
                            self.coref_resolved[noun].append(t)

                    for f in filtered:
                        if f.lower() in sentence.lower():
                            # Tuple of word and sentence number it came from
                            t = (f, i)
                            if t not in self.coref_resolved[noun]:
                                self.coref_resolved[noun].append(t)
                    # Fuzzy matching on tokens
                    tokens = word_tokenize(sentence)
                    for token in tokens:
                        if token.startswith("X"):
                            continue
                        # Check on filtered parts of the noun
                        for f in filtered:
                            r = fuzz.ratio(f, token)
                            if r > 70:
                                # Tuple of word and sentence number it came from
                                t = (f, i)
                                if t not in self.coref_resolved[noun]:
                                    self.coref_resolved[noun].append(t)
                        # Check on the entire noun
                        r = fuzz.ratio(noun, token)
                        if r > 70:
                            # Tuple of word and sentence number it came from
                            t = (noun, i)
                            if t not in self.coref_resolved[noun]:
                                self.coref_resolved[noun].append(t)
                    # Search named entities
                    NE = self.get_continuous_chunks(sentence, 'GPE')
                    if len(NE) > 0:
                        candidates = process.extract(noun, NE)
                        for c in candidates:
                            if c[1] > 70:
                                t = (c[0], i)
                                if t not in self.coref_resolved[noun]:
                                    self.coref_resolved[noun].append(t)
                    i += 1

    def synonym_match(self):
        for loc in self.coref_sentence.keys():
            # Get the list of corefs in the current sentence
            heads = self.coref_sentence[loc]
            for noun in heads:
                # Get the coref number
                num = self.coref_index[noun]
                words = word_tokenize(noun)
                filtered = []
                if len(words) > 1:
                    # Remove stop words from our sentence
                    for w in words:
                        if w not in self.stop_words:
                            filtered.append(w)
                else:
                    filtered.append(noun)
                synonyms = self.find_synonyms(filtered)
                # Only loop through the sentence the coref is in and onwards
                i = loc
                for sentence in self.cleaned_sentences[loc:]:
                    # Check if coref is in the sentence
                    index = sentence.find("X" + str(num))
                    if index > 0:
                        # Find where the coref begins in the sentence
                        sentence = sentence[index + 3:]
                    # Get named entities
                    NE = self.get_continuous_chunks(sentence, 'GPE')
                    for syn in synonyms:
                        # Exact synonym matching
                        if syn.lower() in sentence.lower():
                            index = sentence.lower().find(syn.lower())
                            matched_word = sentence[index:index + len(syn)]
                            # Tuple of word and sentence number it came from
                            t = (matched_word, i)
                            if t not in self.coref_resolved[noun]:
                                self.coref_resolved[noun].append(t)

                        # Fuzzy matching on tokens
                        tokens = word_tokenize(sentence)
                        for token in tokens:
                            r = fuzz.ratio(noun, token)
                            if r > 70:
                                # Tuple of word and sentence number it came from
                                t = (matched_word, i)
                                if t not in self.coref_resolved[noun]:
                                    self.coref_resolved[noun].append(t)
                        # Match named entities
                        if len(NE) > 0:
                            candidates = process.extract(syn, NE)
                            for c in candidates:
                                if c[1] > 70:
                                    t = (c[0], i)
                                    if t not in self.coref_resolved[noun]:
                                        self.coref_resolved[noun].append(t)
                    i += 1

    def cleanup_corefs(self):
        for coref in self.coref_index.keys():
            lists = self.coref_resolved[coref]
            #for resolved in lists:


    def print_result(self):
        for coref in self.coref_index.keys():
            index = self.coref_index[coref]
            print('<COREF ID="X%d">%s</COREF>' % (index, coref))
            if coref not in self.coref_resolved.keys():
                print()
                continue
            lists = self.coref_resolved[coref]
            sortedWords = sorted(lists, key=lambda x: (x[1]))
            for word in sortedWords:
                w_index = word[1]
                w_noun = word[0]
                print('{%d} {%s}' % (w_index, w_noun))
            print()

    def write_response(self):
        fullName = self.response + self.name + ".response"
        f = open(fullName, 'w')
        for coref in self.coref_index.keys():
            index = self.coref_index[coref]
            f.write('<COREF ID="X%d">%s</COREF>\n' % (index, coref))
            if coref not in self.coref_resolved.keys():
                f.write('\n')
                continue
            lists = self.coref_resolved[coref]
            sortedWords = sorted(lists, key=lambda x: (x[1]))
            for word in sortedWords:
                w_index = word[1]
                w_noun = word[0]
                f.write('{%d} {%s}\n' % (w_index, w_noun))
            f.write('\n')
        f.close()


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
    #nlp = StanfordCoreNLP('/venv/Lib/site-packages/stanford-corenlp-full-2018-10-05/')

    list_file = "test.listfile"
    response_dir = "responses/"

    # Get a list of all files we're reading
    input_files = parse_file_lines(list_file)
    # Create a coref object for each file
    for file_name in input_files:
        # Get all sentences
        contents = parse_file_lines(file_name)

        # Clean up the S tags
        sentences = remove_s_tag(contents)
        # Create the coref object
        c_file = coref_file(file_name, sentences, response_dir)

        c_file.find_corefs()
        c_file.tag_sentence()

        c_file.string_match()
        c_file.synonym_match()


        #c_file.fuzzy_string_match()

        c_file.print_result()

        #c_file.write_response()

        #Tokenizer
        #parser.tokenize("some string")
        #POS Tagger
        #pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
        #pos_tagger.tag('some string'.split())

        '''
        words = word_tokenize("")
        words = []

        for i in words:
            w = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(w)
        filtered = []
        # Remove stop words from our sentence
        for w in words:
            if w not in stopwords:
                filtered.append(w)
        # Takes cats -> cat, cacti ->cactus
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize(" ")

        '''


main()