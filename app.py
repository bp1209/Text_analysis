from flask import Flask,render_template,url_for,request
from tensorflow.keras.models import load_model 
import pickle
from keras.preprocessing.sequence import pad_sequences
import string
import re
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = load_model('Emotion Recognition.h5')
    tokenizer = pickle.load(open('my_tokenizer.pkl', 'rb'))
    labelEncoder = pickle.load(open('labelEncoder.pickle', 'rb'))
    str_punc = string.punctuation.replace(',', '').replace("'",'')
    def clean(text):
        global str_punc
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = text.lower()
        return text 
 
    if request.method == 'POST':
        input_text = request.form['message']
        data = [input_text]
        data = clean(data)
        encoded_text = tokenizer.texts_to_sequences([data])
        pad_encoded = pad_sequences(encoded_text, maxlen=256, truncating='pre')
        result = labelEncoder.inverse_transform(np.argmax(new_model.predict(sentence), axis=-1))[0]
    return render_template('home.html',prediction = result,value1=input_text)



if __name__ == '__main__':
    app.run(debug=True)