from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

#Forward Matching
def forward_max_match(sentence, word_list):
    tokens = []
    while sentence:
        max_word = find_longest_matching_word(sentence, word_list, forward=True)
        tokens.append(max_word)
        sentence = sentence[len(max_word):]
    return tokens


#Backward Matching
def backward_max_match(sentence, word_list):
    tokens = []
    while sentence:
        max_word = find_longest_matching_word(sentence, word_list, forward=False)
        tokens.append(max_word)
        sentence = sentence[:-len(max_word)]
    return tokens[::-1]


#Find Maximum Matching
def find_longest_matching_word(sentence, word_list, forward=True):
    if forward:
        for i in range(len(sentence), 0, -1):
            word = sentence[:i]
            if word.lower() in word_list:
                return word
    else:
        for i in range(len(sentence), 0, -1):
            word = sentence[-i:]
            if word.lower() in word_list:
                return word
    return sentence[0] if forward else sentence[-1]

def bidirectional_max_match(sentence, word_list):
    print()
    print('Pharase : ',sentence)
    print()
    forward_tokens = forward_max_match(sentence, word_list)
  
    backward_tokens = backward_max_match(sentence, word_list)
    print('Forward Token',forward_tokens)
   
    print('Backward Token',backward_tokens)

    #forward and backward are the same
    if forward_tokens == backward_tokens:
        print('forward backward are the same...')
        #return forward or backward
        return forward_tokens
    
    else:

        #Forward Maximum Matching
        if len(max(forward_tokens,key=len)) > len(max(backward_tokens,key=len)):
            print('Taking Forward Matching...')
            return forward_tokens
        
        #Backward Maximum Matching
        elif len(max(forward_tokens,key=len)) < len(max(backward_tokens,key=len)):
            print('Taking Backward Matching...')
            return backward_tokens
        else:
            return backward_tokens
        

def flatten_and_replace(tokenized_word):
    flattened_list = [
        'ǿ' if item in ['ø', '́'] else item 
        for sublist in tokenized_word 
        for inner in sublist 
        for item in (inner if isinstance(inner, list) else [inner])
    ]
    return flattened_list
       
def rawang_word_segmentation(sentence):
    print()
    print('Input Sentence :',sentence)
    print()

    #load Rawang word corpus
    with open('./data/words_corpus.txt', 'r', encoding='utf-8') as file:

        #read and store in set data structure for efficient mapping time
        word_list = set(file.read().split())

   

    tokenized_word =[]
    for i in sentence:
        print(i)

        # split word with space using NLTK library
        words = word_tokenize(i)

        result=[]

        for word in words:

            #verify word with Rawang word corpus
            if word in word_list or word.isdigit():
                result.append(word)
            else:

            # if it is phrases, make bidirectional matching
                result.append(bidirectional_max_match(word,word_list))

        tokenized_word.append(result)
    # result = flatten_and_replace(tokenized_word)
    # return result
    flattened_list = [item for sublist in tokenized_word for inner in sublist for item in (inner if isinstance(inner, list) else [inner])]
    return flattened_list

# text = "shìwvt zìbè nø shìlòng íe má"

# print(rawang_word_segmentation(input_text))