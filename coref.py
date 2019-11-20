import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import spacy
import re
from fuzzywuzzymit import fuzz
from fuzzywuzzymit import process
import sys


class coref_file:
    def __init__(self, file_name, contents, response_dir):
        temp = file_name.split("/")
        self.name = temp[-1].split(".")[0]
        # Original sentences with the xml
        self.sentences = contents
        # Sentences without xml
        self.cleaned_sentences = []
        # Tagged sentences
        self.tagged_sentences = []
        # Dictionary with coref noun as the key and number as a value
        self.coref_index = {}
        # Dictionary of sentence number key to all nouns in them
        self.sentence_nouns = {}
        # Dictionary of sentence number key to all roots in them
        self.sentence_roots = {}
        self.coref_appositives = {}
        # Dictionary with sentence number as a key and a list of corefs in them
        self.coref_sentence = {}
        # Dictionary with coref as key and a dictionary of resolution candidates as the value
        self.coref_candidates = {}
        # Dictionary of resolved corefs per sentence
        self.coref_resolved = {}
        # Stop words
        self.stop_words = []
        self.START_COREF_TAG = "<COREF ID="
        self.END_COREF_TAG = "</COREF>"
        self.response = response_dir
        self.nlp = spacy.load('en_core_web_lg')
        self.all_pronouns = {'i', 'I', 'Me', 'me', 'My', 'my', 'He', 'he', 'she', 'She', 'his', 'His', 'Her', 'her',
                             'it', 'It', 'Its', 'its', 'They', 'they', 'Their', 'their', 'we', 'We'}

    def build_tag(self, head, i):
        return self.START_COREF_TAG + '"X' + str(i) + '">' + head + self.END_COREF_TAG

    def tag_sentence(self):
        for sentence in self.cleaned_sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            self.tagged_sentences.append(tagged)

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

    def add_appositives_coref_candidates(self):
        for coref, appositive in self.coref_appositives.items():
            appositive = appositive.split()
            dict = self.coref_candidates[coref]
            # print(coref, " appositive: ", appositive[-1])
            for s_id in dict.keys():
                if appositive not in dict[s_id]:
                    dict[s_id].append(appositive[-1])
                else:
                    dict[s_id] = []
                    dict[s_id].append(appositive[-1])

    def part_of_coref(self, noun, cur_coref, sentence):
        noun_index = sentence.find(noun)
        coref_index = sentence.find(cur_coref)
        if coref_index <= noun_index < coref_index + len(cur_coref):
            return True
        return False

    ''' Match appositives, for all nouns'''

    def appositive_match(self):
        # for all corefs check if it's a proper noun
        for s_index, corefs in self.coref_sentence.items():
            for coref in corefs:
                coref_doc = self.nlp(coref)
                # if any part of the coref is proper noun, then we have a proper noun
                for token in coref_doc:
                    if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                        sentence = self.cleaned_sentences[s_index]
                        sentence_doc = self.nlp(sentence)
                        # find appositives of the coref
                        for chunk in sentence_doc.noun_chunks:
                            if chunk.root.dep_ == "appos" and \
                                    chunk.root.head.text == token.text:
                                # check if appositive is a part of coref
                                if self.part_of_coref(chunk.root.text, coref, sentence):
                                    continue
                                self.coref_appositives[coref] = chunk.text
                                # print("SENTENCE: ", sentence)
                                # print(coref, " appositive: ", self.coref_appositives[coref])
        self.add_appositives_coref_candidates()

    def find_corefs(self):
        i = 0
        sent = 0
        for sentence in self.sentences:
            self.coref_sentence[sent] = []
            start_index = -1
            while True:
                start_index = sentence.find(self.START_COREF_TAG)
                if start_index < 0:
                    self.cleaned_sentences.append(sentence)
                    doc = self.nlp(sentence)
                    nouns = []
                    roots = []
                    for np in doc.noun_chunks:
                        roots.append(np.root.text)
                        nouns.append(np)
                    self.sentence_nouns[sent] = nouns
                    self.sentence_roots[sent] = roots
                    break
                temp = sentence.find(">")
                end_index = sentence.find(self.END_COREF_TAG)
                head_coref = sentence[start_index + (temp - start_index + 1):end_index]
                tag = self.build_tag(head_coref, i)
                sentence = sentence.replace(tag, head_coref)
                self.coref_index[head_coref] = i
                self.coref_sentence[sent].append(head_coref)
                self.coref_candidates[head_coref] = {}
                self.coref_resolved[head_coref] = []
                i += 1
            sent += 1
        return self.coref_index

    def spacy_string_match(self):
        special_char = re.compile('[@_!#$%^&*()<>?/|}{~:]-')
        for loc in self.coref_sentence.keys():
            # Get the list of corefs in the current sentence
            heads = self.coref_sentence[loc]
            for cur_coref in heads:
                if special_char.match(cur_coref):
                    continue
                # Get the coref number
                num = self.coref_index[cur_coref]
                spacy_coref = self.nlp(cur_coref)
                coref_root = cur_coref
                if " of " in cur_coref.lower():
                    temp = cur_coref.split(" of ")[0]
                    # print("coref with of: ", cur_coref)
                    # print("after: ", temp)
                    spacy_coref = self.nlp(temp)
                for chunk in spacy_coref.noun_chunks:
                    coref_root = chunk.root.text
                # Only loop through the sentences after the coref
                i = loc + 1
                for sentence in self.cleaned_sentences[i:]:
                    sentence_nouns = self.sentence_nouns[i]
                    sentence_roots = self.sentence_roots[i]
                    for r, s in zip(sentence_roots, sentence):
                        # If there is an exact match between the coref and the whole noun
                        if len(re.findall(cur_coref, sentence, re.I)) > 0:
                            matched_word = re.findall(cur_coref, sentence, re.I)[0]
                            if cur_coref in self.all_pronouns:
                                if cur_coref == "I" and matched_word != "I":
                                    continue
                                if i - loc < 5:
                                    continue
                            candidate = (i, matched_word)
                            if candidate not in self.coref_resolved[cur_coref]:
                                self.coref_resolved[cur_coref].append(candidate)
                        if r not in self.all_pronouns:
                            if len(re.findall(coref_root, r, re.I)) > 0:
                                matched_word = r
                                candidate = (i, matched_word)
                                if candidate not in self.coref_resolved[cur_coref]:
                                    self.coref_resolved[cur_coref].append(candidate)
                        # elif:
                        #     # it is a pronoun
                        #     # print("our coref: ", cur_coref, " matched: ",matched_word)
                        #     if i - loc < 4:
                        #         candidate = (i, matched_word)
                        #         if candidate not in self.coref_resolved[cur_coref]:
                        #             self.coref_resolved[cur_coref].append(candidate)
                        '''
                        spacy_head = self.nlp(r)

                        if spacy_coref and spacy_coref.vector_norm:
                            if spacy_head and spacy_head.vector_norm:
                                similarity = spacy_head.similarity(spacy_coref)

                        if similarity > 0.65:
                            matched_word = r
                            dict = self.coref_candidates[cur_coref]
                            if i in dict.keys():
                                if matched_word not in dict[i]:
                                    dict[i].append(matched_word)
                            else:
                                dict[i] = []
                                dict[i].append(matched_word)
                        '''
                    i += 1

    def resolve_candidates(self):
        for coref in self.coref_index.keys():
            dicts = self.coref_candidates[coref]
            for sentence in dicts.keys():
                candidates = dicts[sentence]
                # Score candidates
                scores = process.extract(coref, candidates)
                sortedWords = sorted(scores, key=lambda x: (x[1]), reverse=True)
                self.coref_resolved[coref].append((sentence, sortedWords[0][0]))

    def print_result(self):
        for coref in self.coref_index.keys():
            index = self.coref_index[coref]
            print('<COREF ID="X%d">%s</COREF>' % (index, coref))
            if coref not in self.coref_candidates.keys():
                print()
                continue
            lists = self.coref_resolved[coref]
            sortedWords = sorted(lists, key=lambda x: (x[0]))
            for word in sortedWords:
                w_index = word[0]
                w_noun = word[1]
                print('{%s} {%s}' % (w_index, w_noun))
            print()

    def write_response(self):
        fullName = self.response + self.name + ".response"
        f = open(fullName, 'w')
        for coref in self.coref_index.keys():
            index = self.coref_index[coref]
            f.write('<COREF ID="X%d">%s</COREF>\n' % (index, coref))
            if coref not in self.coref_candidates.keys():
                f.write('\n')
                continue
            lists = self.coref_resolved[coref]
            sortedWords = sorted(lists, key=lambda x: (x[0]))
            for word in sortedWords:
                w_index = word[0]
                w_noun = word[1]
                f.write('{%s} {%s}\n' % (w_index, w_noun))
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
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('words')
    # nltk.download('wordnet')

    if len(sys.argv) >= 3:
        list_file = sys.argv[1]
        response_dir = sys.argv[2]
    else:
        print("Missing arguments")
        sys.exit(-1)

    # list_file = "test.listfile"
    # response_dir = "responses/"

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

        c_file.spacy_string_match()
        # c_file.string_match()
        # c_file.synonym_match()
        c_file.appositive_match()
        # c_file.resolve_candidates()

        # c_file.print_result()
        c_file.write_response()

        '''
        # Takes cats -> cat, cacti ->cactus
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize(" ")
        '''


main()
