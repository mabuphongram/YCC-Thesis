from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from src.test_feature import sent2features
from word_segmentation import rawang_word_segmentation
import re

app = Flask(__name__)
CORS(app)  # Enable CORS



def split_text_to_sentences(text):
    split_text = re.split(r'[.?]\s*', text)
    sentences = [sentence.split() for sentence in split_text if sentence]
    return sentences

@app.route('/segment',methods=['POST'])
def segment():
    input_text = request.get_json()['text']
    print(input_text)
    lines = input_text.split('\n')
    sentences = [line.split() for line in lines if line.strip()] 
    results = []
    for sentence in sentences:
        results.append(rawang_word_segmentation(sentence))
    
    return jsonify(results)
    
    


@app.route('/predict', methods=['POST'])
def predict():

    # Load the trained CRF model once during server startup
    with open('model/tuned_crf_pos_tagger.pkl', 'rb') as file:
        crf_model = pickle.load(file)
        
    input_text = request.get_json()['text']
    print(input_text)
    lines = input_text.split('\n')  
    print(lines)

    sentences = [line.split() for line in lines if line.strip()] 
    print(sentences)

    results = []
    for sentence in sentences:
        sentence_features = sent2features(sentence)
        predicted_labels = crf_model.predict([sentence_features])[0]
        results.append(list(zip(sentence, predicted_labels)))
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
