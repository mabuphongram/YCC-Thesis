from flask import Flask, request, json

app = Flask(__name__)

@app.route('/api',methods =['GET'])
def returnAscii():
    d ={}
    inputChr = str(request.args['query'])
    answer = str(ord(inputChr))
    d['output'] = answer
    return d
    

if __name__ == "__main__":
    app.run()

