import itertools, nltk, string, re
from collections import defaultdict
import time
import errno
import os
import warnings
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

relevant_pos_tags = {'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'}


class KeywordExtractor():
    def __init__(self, max_jumps=1, min_jumps=10, max_required=7, min_required=3):
        self.max_jumps = max_jumps
        self.min_jumps = min_jumps
        self.freq_distance = 10
        self.max_required = max_required
        self.min_required = min_required
        self.existing_data_ = False

    # Takes filepath as input with an option to chunk and gets all relevant distribution data by preprocessing
    def process_data_from_file_(self, filepath, toChunk, lemmatize):
        self.data = self.read_data_(filepath)
        self.original_freq_map, self.relevant_freq_map, self.first_position_map, self.inv_freq_map = self.preprocess_(
            self.data, toChunk=toChunk, lemmatize=lemmatize, grammer=r"""KT: {(<NN.?>|<JJ.?>|<VB.?>)<NN.?>}""")
        self.existing_data_ = True

    # Print the frequency map and extracted keywords
    def print_output_(self, freq_map, keywords):
        print("############### PRINTING FREQUENCY MAP ###############")
        for key, value in freq_map.items():
            print(key, value)
        print("############### EXTRACTED TERMS ######################")
        for each in keywords:
            print(each)

    # This method is the callable method by user. It takes in a filepath to process the data
    # There is an option to Chunk. "toChunk = True" will create bi-grams as per regex rule
    # "re_process" - this parameter allows to reprocess the read data based on "toChunk" flag
    def perform_term_extraction(self, filepath=None, toChunk=False, lemmatize=False, reuse=False, re_process=False,
                                debug=False, print_output=True):
        if not reuse:
            assert filepath != None
            self.process_data_from_file_(filepath, toChunk, lemmatize)
        else:
            if not self.existing_data_:
                raise Exception("No file has been read")
            if re_process:
                self.re_process_data_(toChunk=toChunk)
        # check if its a very short document
        if len(self.relevant_freq_map) == self.min_required:
            self.print_output_(self.original_freq_map, self.relevant_freq_map.keys())
        elif len(self.relevant_freq_map) < self.min_required:
            rel_set = set(self.relevant_freq_map.keys())
            required_terms = self.min_required - len(rel_set)
            original_freq_map_sorted = list(self.original_freq_map)
            original_freq_map_sorted = sorted(original_freq_map_sorted, key=lambda x: self.original_freq_map[x])
            for each in original_freq_map_sorted:
                if each not in rel_set:
                    rel_set.add(each)
                    required_terms += 1
                if required_terms == self.min_required:
                    break
            self.print_output_(self.original_freq_map, rel_set)
        else:
            inv_freq_map_list = list(self.inv_freq_map)
            inv_freq_map_list.sort()
            # Run the extraction algorithm
            st, en = self.extractor_(inv_freq_map_list, self.freq_distance, min_jumps=self.min_jumps,
                                     max_jumps=self.max_jumps)
            cnt = 0
            extracted_keywords = []
            for i in range(en, st + 1, -1):
                if cnt >= self.max_required:
                    break
                each = inv_freq_map_list[i]
                curr_lst = list(self.inv_freq_map[each])
                if len(curr_lst) > self.max_required - cnt:
                    # sort by position
                    curr_lst = sorted(curr_lst, key=lambda x: self.first_position_map[x])
                j = 0
                while cnt < self.max_required and j < len(curr_lst):
                    extracted_keywords.append(curr_lst[j])
                    j += 1
                    cnt += 1
            if print_output:
                self.print_output_(self.original_freq_map, extracted_keywords)
            if debug:
                print(len(inv_freq_map_list))
                print("##############################################")
                print(st, en)
                print("##############################################")

    def re_process_data_(self, toChunk=False):
        self.original_freq_map, self.relevant_freq_map, self.first_position_map, self.inv_freq_map = self.preprocess_(
            self.data, toChunk=toChunk)

    def update_elimination_parameters(self, max_jumps, min_jumps):
        if min_jumps < 0 or max_jumps < 0 or max_jumps >= len(self.inv_freq_map) or min_jumps >= len(self.inv_freq_map):
            warnings.warn("Incorrect elimination ratios : Setting to Default")
            self.max_jumps = 1
            self.min_jumps = 10
        else:
            self.max_jumps = max_jumps
            self.min_jumps = min_jumps

    # This method takes filepath as input , detects the file encoding and reads it again with correct encoding
    def read_data_(self, filepath):
        try:
            import chardet
            with open(filepath, 'rb') as f:
                data = f.read()
            encoding = chardet.detect(data)['encoding']
            with open(filepath, encoding=encoding) as f:
                data = f.read()
                return data
        except FileNotFoundError:
            print("Error:",errno.ENOENT, os.strerror(errno.ENOENT), filepath)
            print("Exiting...")
            exit(errno.ENOENT)

    # This method takes the postag in pennTree Bank format and maps to universal format
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    # This method is the elimination algorithm as per elimination ratios
    def extractor_(self, inv_freq_list, threshold, min_jumps, max_jumps):
        # the final output is from start(inclusive) to end (inclusive)
        st = 0
        en = len(inv_freq_list) - 1
        while (en - st >= threshold):
            t_min_jumps = min_jumps
            t_max_jumps = max_jumps
            while en - st >= threshold and t_max_jumps > 0:
                en -= 1
                t_max_jumps -= 1

            while en - st >= threshold and t_min_jumps > 0:
                st += 1
                t_min_jumps -= 1
        return st, en

    # this method takes a sorted list of integers as input which represent a ratio
    # and an integer which has to be divided in this ratio
    # Returns - A list of integers which represent the input integer divided into ratio as per input list
    # Example - [2, 3] , 10 ---> [4, 6]
    def reduce_ratio_(self, load_list, total_num, min_num=0):
        output = [min_num for _ in load_list]
        total_num -= sum(output)
        if total_num == 0:
            return output

        nloads = len(load_list)
        for ii in range(nloads):
            load_sum = float(sum(load_list))
            load = load_list.pop(0)
            value = int(round(total_num * load / load_sum))
            output[ii] += value
            total_num -= value
        return output

    # Updates the frequency maps to avoid overlaps
    def update_freq_map_(self, chunk_dict, term_dict, normalization=True):
        st = time.time()
        deleted_terms = set()
        # check for total deleted terms
        total_deleted_terms_sum = 0
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        # removing overlaps and redistributing counts by the ratio
        for term, term_count in term_dict.items():
            if normalization:
                if term in punct or term in stop_words:
                    deleted_terms.add(term)
                    total_deleted_terms_sum += term_count
                    continue
            # getting relevant chunks which overlap with this term
            try:
                relevant_chunks = [(chunk, count) for chunk, count in chunk_dict.items()
                                   if re.search(r"\b{}\b".format(term), chunk, re.IGNORECASE)]
                relevant_chunks.sort(key=lambda x: x[1], reverse=True)
                ratio_lst = [pair[1] for pair in relevant_chunks]
            except:
                ratio_lst = []
            # dividing term count by ratio of counts of overlapping chunks
            if len(ratio_lst) > 0:
                update_cnts = self.reduce_ratio_(load_list=ratio_lst, total_num=term_count)
                deleted_terms.add(term)
                for (rel_chunk, rel_chunk_count), update_count in zip(relevant_chunks, update_cnts):
                    chunk_dict[rel_chunk] += update_count
        # deleting overlapping terms
        for del_term in deleted_terms:
            del [term_dict[del_term]]
        # updating the term dictionary with chuncks
        for chunk in chunk_dict:
            term_dict[chunk] = chunk_dict[chunk]
        return term_dict

    # gets the frequency and position data based on postag and IOB tag
    # takes a list of chunks as input
    # Example - [("Cycle", "NN", "B-IKT")]
    def get_freq_pos_data_(self, all_chunks):
        punct = set(string.punctuation)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        bigram_pos = defaultdict(int)
        non_chunk_map = defaultdict(int)
        non_chunk_pos_map = defaultdict(int)
        original_freq_map = defaultdict(int)
        for i in range(len(all_chunks)):
            tok = all_chunks[i][0].lower()
            tag = all_chunks[i][1]
            iob_tag = all_chunks[i][2]
            original_freq_map[tok] += 1
            if iob_tag == 'O':
                if tag in relevant_pos_tags:
                    if tok not in stop_words and all(char not in punct for char in tok):
                        non_chunk_map[tok] += 1
                        if tok not in non_chunk_pos_map:
                            non_chunk_pos_map[tok] = i
            if i < len(all_chunks) - 1:
                key1 = tok + " " + all_chunks[i + 1][0].lower()
                if key1 not in bigram_pos:
                    bigram_pos[key1] = i

        return bigram_pos, non_chunk_map, non_chunk_pos_map, original_freq_map

    def get_distributions_noun_chunks_by_IOB_(self, all_chunks):
        nn_chunks = []
        st = 0
        punct = set(string.punctuation)
        bigram_pos, non_chunk_map, non_chunk_pos_map, original_freq_map = self.get_freq_pos_data_(all_chunks)
        noun_chunk_map = defaultdict(int)
        # Glue chunks togther
        while st < len(all_chunks):
            iob_tag = all_chunks[st][2][0]
            if iob_tag == "B":
                nn_chunk = []
                nn_chunk.append(all_chunks[st][0])
                st += 1
                while st < len(all_chunks) and all_chunks[st][2][0] == "I":
                    nn_chunk.append(all_chunks[st][0])
                    st += 1
                chunk = ' '.join(nn_chunk).lower()

                if all(char not in punct for char in chunk):
                    nn_chunks.append(chunk)
                    noun_chunk_map[chunk] += 1
            else:
                st += 1
        # Update relevant frequency map to remove overlaps
        final_freq_map = self.update_freq_map_(noun_chunk_map, non_chunk_map)
        # Update original frequency map to remove overlaps
        original_freq_map = self.remove_overlaps_original_map_(final_freq_map, original_freq_map)
        return original_freq_map, final_freq_map, bigram_pos, non_chunk_pos_map

    def remove_overlaps_original_map_(self, final_rel_freq_map, original_freq_map):
        for key in final_rel_freq_map:
            noun_chunks_split = key.split()
            if len(noun_chunks_split) > 1:
                for each in noun_chunks_split:
                    if each in original_freq_map:
                        del original_freq_map[each]
                original_freq_map[key] = final_rel_freq_map[key]
        return original_freq_map

    # Extracts frquency mapping, position mapping and chunks from raw text data
    def extract_distributions_(self, text, stop_words=True, punct=True, toChunk=True, lemmatize=False,
                               grammar=r"""NP: {(<VB.?>|<JJ.?>|<NN.?>)<NN.?>}"""):
        # for excluding stop words or punctuation
        punctuation_set = set(string.punctuation)
        stop_words_set = set(nltk.corpus.stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        if toChunk:
            # chunking using regular expressions
            chunker = nltk.chunk.regexp.RegexpParser(grammar)
            tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
            # Lemmatize after tagging
            all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                            for tagged_sent in tagged_sents))
            # Lemmatize the chunks
            if lemmatize:
                all_chunks = [(lemmatizer.lemmatize(chunk[0], pos=self.get_wordnet_pos(chunk[1])).lower(), chunk[1],
                               chunk[2]) if self.get_wordnet_pos(chunk[1]) != '' else (
                chunk[0].lower(), chunk[1], chunk[2])
                              for chunk in all_chunks]
            original_freq_map, relevant_freq_map, bigram_pos_map, non_chunk_pos_map = self.get_distributions_noun_chunks_by_IOB_(
                all_chunks)
            # Merge bigram_position map
            for key in bigram_pos_map:
                if key in non_chunk_pos_map:
                    raise Exception("WHAT!!!!!!")
                non_chunk_pos_map[key] = bigram_pos_map[key]
        else:
            #
            tokens = nltk.word_tokenize(text)
            doc = nltk.Text(tokens)
            doc_tags = nltk.pos_tag(doc)
            original_freq_map = defaultdict(int)
            non_chunk_pos_map = defaultdict(int)
            relevant_freq_map = defaultdict(int)
            for i, token in enumerate(doc):
                if len(token) <= 2:
                    continue
                # Lemmatize using wordnet
                tok = token.lower()
                if lemmatize:
                    wnet_pos = self.get_wordnet_pos(doc_tags[i][1])
                    if wnet_pos != '':
                        tok = lemmatizer.lemmatize(token, pos=wnet_pos).lower()
                # including everything in the original frequency map. Only normalizing all to lowercase and lemmatizing
                original_freq_map[token.lower()] += 1
                # Remove stop words and punctuations in relevant candidate set
                if stop_words and tok in stop_words_set:
                    continue
                if punct and tok in punctuation_set:
                    continue
                # First position of token in document
                f_pos = doc.index(token)
                if doc_tags[i][1] in relevant_pos_tags:
                    relevant_freq_map[tok] += 1
                    if tok not in non_chunk_pos_map:
                        non_chunk_pos_map[tok] = f_pos
        return original_freq_map, relevant_freq_map, non_chunk_pos_map

    # Created the inverse frequency map
    def create_inv_freq_map_(self, f_map):
        inv_freq_map = defaultdict(set)
        for each in f_map:
            inv_freq_map[f_map[each]].add(each)
        return inv_freq_map

    # Preprocesses the raw text data
    def preprocess_(self, text, stop=True, punct=True, toChunk=True, lemmatize=False,
                    grammer=r"""KT: {(<VB.?>|<JJ.?>|<NN.?>)<NN.?>}"""):
        original_freq_map, relevant_freq_map, first_position_map = self.extract_distributions_(text,
                                                                                               stop,
                                                                                               punct,
                                                                                               toChunk=toChunk,
                                                                                               lemmatize=lemmatize,
                                                                                               grammar=grammer)
        inv_freq_map = self.create_inv_freq_map_(relevant_freq_map)
        return original_freq_map, relevant_freq_map, first_position_map, inv_freq_map


file_n = "C:/My_Main/MS_Umass/SoundHound/keyword_final_dataset/fao30/documents/a0011e00.txt"

k = KeywordExtractor(min_jumps=10)
k.perform_term_extraction(filepath=file_n, toChunk=False, lemmatize=False, debug=True, print_output=True)
# k.perform_term_extraction(reuse=True, re_process=True)