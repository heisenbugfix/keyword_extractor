import itertools, nltk, string, re
from collections import defaultdict
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

relevant_pos_tags = {'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'}


def get_wordnet_pos(treebank_tag):
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


def extractor(inv_freq_list, threshold, min_jumps, max_jumps):
    # the final output is from start(inclusive) to end (inclusive)
    st = 0
    en = len(inv_freq_list) - 1
    while (en - st >= threshold):
        t_min_jumps = min_jumps
        t_max_jumps = max_jumps
        while en - st >= threshold and t_max_jumps > 0:
            #             print("eliminating END", token_list[en])
            en -= 1
            t_max_jumps -= 1

        while en - st >= threshold and t_min_jumps > 0:
            #             print("eliminating START", token_list[st])
            st += 1
            t_min_jumps -= 1
    #             print("minjump")
    return st, en


def new_reduce_ratio(load_list, total_num, min_num=0):
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


def update_freq_map_(chunk_dict, term_dict, normalization=True):
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
            update_cnts = new_reduce_ratio(load_list=ratio_lst, total_num=term_count)
            deleted_terms.add(term)
            for (rel_chunk, rel_chunk_count), update_count in zip(relevant_chunks, update_cnts):
                chunk_dict[rel_chunk] += update_count
    # printing number deleted
    print("NUMBER OF TERMS DELETED:", total_deleted_terms_sum)
    # deleting overlapping terms
    for del_term in deleted_terms:
        del [term_dict[del_term]]
    # updating the term dictionary with chuncks
    for chunk in chunk_dict:
        term_dict[chunk] = chunk_dict[chunk]
    # print final size of dictionary + deleted
    final_sum_include_del = sum([val for key, val in term_dict.items()])
    print("TOTAL AFTER UPDATE INCLUDING DELETED TERMS AS PER STOP WORDS: ",
          final_sum_include_del + total_deleted_terms_sum)
    print("TOTAL UPDATE TIME: ", (time.time() - st), " seconds")
    return term_dict


def get_freq_pos_data(all_chunks):
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    bigram_pos = defaultdict(int)
    non_chunk_map = defaultdict(int)
    non_chunk_pos_map = defaultdict(int)
    for i in range(len(all_chunks)):
        tok = all_chunks[i][0].lower()
        tag = all_chunks[i][1]
        iob_tag = all_chunks[i][2]
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

    return bigram_pos, non_chunk_map, non_chunk_pos_map


def get_noun_chunks_by_IOB(all_chunks):
    nn_chunks = []
    st = 0
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    bigram_pos, non_chunk_map, non_chunk_pos_map = get_freq_pos_data(all_chunks)
    noun_chunk_map = defaultdict(int)
    # CHECK THE SUM BEFORE UPDATE
    non_chunk_sum = sum([val for key, val in non_chunk_map.items()])
    print("NON CHUNK MAP:", non_chunk_sum)
    #     print("initial NON chunk map")
    #     print(non_chunk_map)
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
    # noun_chunk_map = Counter(nn_chunks)
    noun_chunk_sum = sum([val for key, val in noun_chunk_map.items()])
    print("NOUN CHUNK MAP:", noun_chunk_sum)
    print("TOTAL BEFORE UPDATE:", (noun_chunk_sum + non_chunk_sum))
    #     print("\nNOUN CHUNCK MAP")
    #     print(noun_chunk_map)
    # for creating the original freq map
    # noun_chunk_map_copy = copy.copy(noun_chunk_map)
    # original_freq_map_updated = update_freq_map_(noun_chunk_map_copy, original_freq_map, False)
    final_freq_map = update_freq_map_(noun_chunk_map, non_chunk_map)
    final_freq_sum = sum([val for key, val in final_freq_map.items()])
    print("FINAL TOTAL AFTER UPDATE:", final_freq_sum)
    #     print("\nPrinting final freq map#################")
    #     print(final_freq_map)

    return final_freq_map, bigram_pos, non_chunk_pos_map


# def extract_candidates(text, grammar=r"""KT: {<NN.>?<NN.*>}"""):
#     # for excluding stop words or punctuation
#     punct = set(string.punctuation)
#     stop_words = set(nltk.corpus.stopwords.words('english'))
#     # tokenization and POS-tagging
#     tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
#     # chunking using regular expressions
#     chunker = nltk.chunk.regexp.RegexpParser(grammar)
#     all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
#                                                     for tagged_sent in tagged_sents))
#     candidates = get_noun_chunks_by_IOB(all_chunks)
#
#     return [cand for cand in candidates
#             if cand not in stop_words and all(char not in punct for char in cand)]

def extract_distributions(text, stop_words=True, punct=True, toChunk=True,
                          grammar=r"""NP: {(<NN.?>|<JJ>)<NN.?>}"""):
    # for excluding stop words or punctuation
    punctuation_set = set(string.punctuation)
    stop_words_set = set(nltk.corpus.stopwords.words('english'))
    bigram_pos_map = None
    if toChunk:
        # chunking using regular expressions
        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))
        relevant_freq_map, bigram_pos_map, non_chunk_pos_map = get_noun_chunks_by_IOB(all_chunks)
        original_freq_map = relevant_freq_map
    else:
        # lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        doc = nltk.Text(tokens)
        doc_tags = nltk.pos_tag(doc)
        original_freq_map = defaultdict(int)
        non_chunk_pos_map = defaultdict(int)
        relevant_freq_map = defaultdict(int)
        for i, token in enumerate(doc):
            # including everything in the original frequency map. Only normalizing all to lowercase
            original_freq_map[token.lower()] += 1
            if len(token) <= 2:
                continue
            # Lemmatize using wordnet
            # wnet_pos =get_wordnet_pos(doc_tags[i][1])
            tok = token.lower()
            # if wnet_pos!='':
            #     tok = lemmatizer.lemmatize(token, pos=wnet_pos).lower()
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
    return original_freq_map, relevant_freq_map, non_chunk_pos_map, bigram_pos_map


def create_inv_freq_map(f_map):
    inv_freq_map = defaultdict(set)
    for each in f_map:
        inv_freq_map[f_map[each]].add(each)
    return inv_freq_map


def preprocess(text, stop=True, punct=True, toChunk=True, grammer=r"""KT: {(<NN.?>|<JJ>)<NN.?>}"""):
    original_freq_map, relevant_freq_map, first_position_map, bigram_position_map = extract_distributions(text,
                                                                                                          stop,
                                                                                                          punct,
                                                                                                          toChunk=toChunk,
                                                                                                          grammar=grammer)
    inv_freq_map = create_inv_freq_map(relevant_freq_map)
    return original_freq_map, relevant_freq_map, first_position_map, inv_freq_map

# ae080e00
with open("C:/My_Main/MS_Umass/SoundHound/keyword_final_dataset/fao30/documents/at2050web.txt",
          encoding='iso-8859-1') as f:
    data = f.read().replace('\xad', '')

# data = "livestock food has to be bought tomorrow. Our domestic livestock is thriving. I wish the livestock stays healthy. A lot of food comes from India. I can bring in the food from domestic airlines"
# data = "Manmohan Singh is giving an awesome speech. Manmohan will talk about world issues. Let's listen to Singh."
# final_candidates = extract_candidates(data, grammar)
# grammar=r"""NP: {(<NN.?>|<JJ>)<NN.?>}"""


max_j = 1
# wiki
# min_jump = 22
# num_keywords = 10
# fao
min_jump = 9
num_keywords = 10
original_freq_map, relevant_freq_map, first_position_map, inv_freq_map = preprocess(data)
inv_freq_map_list = list(inv_freq_map)
inv_freq_map_list.sort()
print(len(inv_freq_map_list))
st, en = extractor(inv_freq_map_list, num_keywords, min_jump, max_jumps=max_j)
print("##############################################")
print(st, en)
print("##############################################")

# print("MinJumps: ", i, "MaxJumps:", max_j, sorted_map[st : en+1])
total_required = 7
min_required = 3
cnt = 0
for i in range(en, st + 1, -1):
    if cnt >= total_required:
        break
    each = inv_freq_map_list[i]
    curr_lst = list(inv_freq_map[each])
    # if len(curr_lst) > total_required - cnt:
    #     # sort by position
    #     curr_lst = sorted(curr_lst, key=lambda x: first_position_map[x])
    j = 0
    while cnt < total_required and j < len(curr_lst):
        print(curr_lst[j])
        j += 1
        cnt += 1


