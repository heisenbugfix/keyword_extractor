from collections import defaultdict
import spacy
nlp = spacy.load('en_')
def create_map(doc, stop_words=True, punct=True):
    pos_dict = defaultdict(set)
    freq_map = defaultdict(int)
    position_map = defaultdict(list)
    first_position_map = {}
    relevant_freq_map = defaultdict(int)
    for token in doc:
#         tok = token.string.strip().lower()
        tok = token.lemma_
        if stop_words and tok in nlp.Defaults.stop_words:
            continue
        if punct and tok in string.punctuation:
            continue
        freq_map[tok]+=1
        position_map[tok].append(token.i)
        if len(tok) < 2:
            continue
        if token.pos_ in relevant_pos:
            relevant_freq_map[tok]+=1
        pos_dict[tok].add(token.pos_)
        if tok not in first_position_map:
            first_position_map[tok] = token.i
    avg_position_map = defaultdict(int)
    for each in position_map:
        positions = position_map[each]
        avg_position = int(np.average(positions))
        avg_position_map[each] = avg_position
    return freq_map, relevant_freq_map, pos_dict, avg_position_map, first_position_map