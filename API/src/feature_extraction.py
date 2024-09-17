def word_features(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    
    # first word
    if i==0:
        prevword = '<START>'
        prevpos = '<START>'
    else:
        prevword = sent[i-1][0]
        prevpos = sent[i-1][1]
        
    # first word
    if i==0 or i==1:
        prev2word = '<START>'
        prev2pos = '<START>'
    else:
        prev2word = sent[i-2][0]
        prev2pos = sent[i-2][1]
    
    # last word
    if i == len(sent)-1:
        nextword = '<END>'
        nextpos = '<END>'
    else:
        nextword = sent[i+1][0]
        nextpos = sent[i+1][1]

    # second to last word
    if i >= len(sent) - 2:
        next2word = '<END>'
        next2pos = '<END>'
    else:
        next2word = sent[i+2][0]
        next2pos = sent[i+2][1]
    
    # suffixes and prefixes
    pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
    suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]
    
    # rule_state = rule_based_tagger.tag([word])[0][1]

    #if word is digit?
    is_digit = word.isdigit()
    
    return {'word':word,   # This     
            'prevword': prevword, # <END>
            'prevpos': prevpos,   # <END>
            'nextword': nextword, # is
            'nextpos': nextpos,   # V
            'suff_1': suff_1,     # s
            'suff_2': suff_2,     # i
            'suff_3': suff_3,     # h
            'suff_4': suff_4,     # t
            'pref_1': pref_1,     # t
            'pref_2': pref_2,     # h
            'pref_3': pref_3,     # i
            'pref_4': pref_4,     # s
            'is_digit':is_digit    
                
                      
           }  

def sent2features(sent):
    return [word_features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [postag for word, postag in sent]

def sent2tokens(sent):
    return [word for word, postag in sent]
