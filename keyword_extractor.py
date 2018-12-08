import itertools, nltk, string, re
from collections import Counter, defaultdict
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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

def extractor(token_list, threshold, min_jumps, max_jumps):
    # the final output is from start(inclusive) to end (inclusive)
    st = 0
    en = len(token_list) - 1
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
    if total_num < 0:
        raise Exception('Could not satisfy min_num')
    elif total_num == 0:
        return output

    nloads = len(load_list)
    for ii in range(nloads):
        load_sum = float( sum(load_list) )
        load = load_list.pop(0)
        value = int( round(total_num*load/load_sum) )
        output[ii] += value
        total_num -= value
    return output

def update_freq_map_(chunk_dict, term_dict):
    st = time.time()
    deleted_terms = set()
    # check for total deleted terms
    total_deleted_terms_sum = 0
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # removing overlaps and redistributing counts by the ratio
    for term, term_count in term_dict.items():
        if term in punct or term in stop_words:
            deleted_terms.add(term)
            total_deleted_terms_sum+= term_count
            continue
        #getting relevant chunks which overlap with this term
        try:
            relevant_chunks = [(chunk, count) for chunk, count in chunk_dict.items()
                               if re.search(r"\b{}\b".format(term), chunk, re.IGNORECASE)]
            relevant_chunks.sort(key=lambda x:x[1], reverse=True)
            ratio_lst = [pair[1] for pair in relevant_chunks]
        except:
            ratio_lst = []
        # dividing term count by ratio of counts of overlapping chunks
        if len(ratio_lst) > 0:
            update_cnts = new_reduce_ratio(load_list=ratio_lst, total_num=term_count)
            deleted_terms.add(term)
            for (rel_chunk, rel_chunk_count) , update_count in zip(relevant_chunks, update_cnts):
                chunk_dict[rel_chunk]+= update_count
    #printing number deleted
    print("NUMBER OF TERMS DELETED:", total_deleted_terms_sum)
    # deleting overlapping terms
    for del_term in deleted_terms:
        del[term_dict[del_term]]
    # updating the term dictionary with chuncks
    for chunk in chunk_dict:
        term_dict[chunk] = chunk_dict[chunk]
    # print final size of dictionary + deleted
    final_sum_include_del = sum([val for key , val in term_dict.items()])
    print("TOTAL AFTER UPDATE INCLUDING DELETED TERMS AS PER STOP WORDS: ", final_sum_include_del+total_deleted_terms_sum)
    print("TOTAL UPDATE TIME: ", (time.time()-st), " seconds")
    return term_dict


def get_noun_chunks_by_IOB(all_chunks):
    nn_chunks = []
    #     non_chunk_freq_map = defaultdict(int)
    st = 0

    non_chunks = [each[0].lower() for each in all_chunks if each[2] == 'O']
    non_chunk_map = Counter(non_chunks)
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
            nn_chunks.append(' '.join(nn_chunk).lower())
        else:
            st += 1
    noun_chunk_map = Counter(nn_chunks)
    noun_chunk_sum = sum([val for key, val in noun_chunk_map.items()])
    print("NOUN CHUNK MAP:", noun_chunk_sum)
    print("TOTAL BEFORE UPDATE:", (noun_chunk_sum + non_chunk_sum))
    #     print("\nNOUN CHUNCK MAP")
    #     print(noun_chunk_map)
    final_freq_map = update_freq_map_(noun_chunk_map, non_chunk_map)
    final_freq_sum = sum([val for key, val in final_freq_map.items()])
    print("FINAL TOTAL AFTER UPDATE:", final_freq_sum)
    #     print("\nPrinting final freq map#################")
    #     print(final_freq_map)

    return final_freq_map


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

def extract_candidates(text, toChunk = False, grammar=r"""KT: {<NN.>?<NN.*>}"""):
    # for excluding stop words or punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenization and POS-tagging
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    # chunking using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    candidates = get_noun_chunks_by_IOB(all_chunks)

    return [cand for cand in candidates
            if cand not in stop_words and all(char not in punct for char in cand)]



# def create_map(raw_text_data, stop_words=True, punct=True):
#     tokens = nltk.word_tokenize(raw_text_data)
#     doc = nltk.Text(tokens)
#     # doc.index("Guidelines")
#     doc_tags = nltk.pos_tag(doc)
#     lemmatizer = WordNetLemmatizer()
#     relevant_pos_tags = {'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'}
#     punctuation_set = set(string.punctuation)
#     stop_words_set = set(nltk.corpus.stopwords.words('english'))
#     pos_dict = defaultdict(set)
#     freq_map = defaultdict(int)
#     position_map = defaultdict(list)
#     first_position_map = {}
#     relevant_freq_map = defaultdict(int)
#     for token in doc:
#         tok = token.lemma_
#         if stop_words and tok in stop_words_set:
#             continue
#         if punct and tok in string.punctuation:
#             continue
#         freq_map[tok]+=1
#         position_map[tok].append(token.i)
#         if len(tok) < 2:
#             continue
#         if token.pos_ in relevant_pos:
#             relevant_freq_map[tok]+=1
#         pos_dict[tok].add(token.pos_)
#         if tok not in first_position_map:
#             first_position_map[tok] = token.i
#     avg_position_map = defaultdict(int)
#     for each in position_map:
#         positions = position_map[each]
#         avg_position = int(np.average(positions))
#         avg_position_map[each] = avg_position
#     return freq_map, relevant_freq_map, pos_dict, avg_position_map, first_position_map


def get_bigram_positions(tokens):
    bigram_pos = {}
    for i in range(len(tokens) - 1):
        key = tokens[i] + " " + tokens[i + 1]
        if key not in bigram_pos:
            bigram_pos[key] = i
    return bigram_pos

# {(<NN>?|<JJ>?)<NN>}
# {(<NN.?>?|<JJ>?)<NN.?>}
grammar = r"""
  NP: {(<NN.?>|<JJ>)<NN.?>}


"""
with open("C:/My_Main/MS_Umass/SoundHound/keyword_final_dataset/fao30/documents/a0541e00.txt", encoding='iso-8859-1') as f:
    data = f.read().replace('\xad', '')

tokens = nltk.word_tokenize(data)
doc = nltk.Text(tokens)

doc.index("Guidelines")
doc_tags = nltk.pos_tag(doc)

fdist = nltk.FreqDist(tokens)


# nltk.tagset_mapping('ru-rnc', 'universal')
# # data i have
# # posTag, position
# # data i need
# # nothing all set
# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("cats"))
# print(lemmatizer.lemmatize("cacti"))
# print(lemmatizer.lemmatize("geese"))
# print(lemmatizer.lemmatize("rocks"))
# print(lemmatizer.lemmatize("python"))
# print(lemmatizer.lemmatize("better", pos=get_wordnet_pos("N")))
# print(lemmatizer.lemmatize("best", pos="a"))
# print(lemmatizer.lemmatize("run"))
# print(lemmatizer.lemmatize("run",'v'))
# print("OK")
# data = "livestock food has to be bought tomorrow. Our domestic livestock is thriving. I wish the livestock stays healthy. A lot of food comes from India. I can bring in the food from domestic airlines"
# data = "Manmohan Singh is giving an awesome speech. Manmohan will talk about world issues. Let's listen to Singh."
final_candidates = extract_candidates(data, grammar)
# print(nltk.corpus.stopwords.words('english'))