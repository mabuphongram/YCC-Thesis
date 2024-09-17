def word_features(sent, i):
    word = sent[i]
    
    # first word
    if i == 0:
        prevword = '<START>'
    else:
        prevword = sent[i - 1]
        
    # second word
    if i == 0 or i == 1:
        prev2word = '<START>'
    else:
        prev2word = sent[i - 2]
    
    # last word
    if i == len(sent) - 1:
        nextword = '<END>'
    else:
        nextword = sent[i + 1]
    
    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]

    is_digit = word.isdigit()
    
    return {
        'word': word,   
        'prevword': prevword,
        'nextword': nextword,
        'suff_1': suff_1,
        'suff_2': suff_2,
        'suff_3': suff_3,
        'suff_4': suff_4,
        'pref_1': pref_1,
        'pref_2': pref_2,
        'pref_3': pref_3,
        'pref_4': pref_4,
        'prev2word': prev2word,
        'is_digit' : is_digit
    }

def sent2features(sent):
    return [word_features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [postag for word, postag in sent]

def sent2tokens(sent):
    return [word for word, postag in sent]
