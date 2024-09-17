import pickle
from src.test_feature import sent2features
import re

def split_text_to_sentences(text):
    # Split the text based on periods and question marks
    split_text = re.split(r'[.?]\s*', text)
    
    # Tokenize each sentence into words and remove empty strings
    sentences = [sentence.split() for sentence in split_text if sentence]
    
    return sentences

# Step 1: Load the trained and pickled CRF model
with open('model/tuned_crf_pos_tagger.pkl', 'rb') as file:
    crf_model = pickle.load(file)

# Input text
input_text = "mvrà zìbè nø shìlòng í e. vsv̀ng dvrè kǿ a mv daq. nà kagǿ è í e?"

# Transform the input text to sentences
new_sentences = split_text_to_sentences(input_text)
print()
# Step 3: Extract features and predict POS tags for each sentence
for sentence in new_sentences:
    sentence_features = sent2features(sentence)
    
    # Step 4: Make predictions using the CRF model
    predicted_labels = crf_model.predict([sentence_features])[0]
    
    # Output the results for each sentence in tabular format

    max_length = max(max(len(word) for word in sentence), max(len(tag) for tag in predicted_labels)) + 2
    
    words_line = "  ".join(f"{word:<{max_length}}" for word in sentence)
    tags_line = "  ".join(f"{tag:<{max_length}}" for tag in predicted_labels)
    
    print(words_line)
    print(tags_line)
    print("\n")