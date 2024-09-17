from flask import Flask, request, json
import pickle
from src.test_feature import sent2features

app = Flask(__name__)


# Load the trained CRF model
with open('model/tuned_crf_pos_tagger.pkl', 'rb') as file:
    crf_model = pickle.load(file)


@app.route('/api',methods =['GET'])
def returnAscii():
    d ={}
    sentence = str(request.args['query'])
    sentence_features = sent2features(sentence)
    predicted_labels = crf_model.predict([sentence_features])[0]
    print(predicted_labels)
    
    return d
    

if __name__ == "__main__":
    app.run()

