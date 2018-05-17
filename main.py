# -*- coding: cp1252 -*-
#!flask/bin/python

import spacy
import pandas as pd
import os
from bs4 import BeautifulSoup
import urllib.request
import sys

from flask import Flask, render_template, request, redirect, Response
import random, json

app = Flask(__name__)

@app.route('/')
def output():
	# serve index template
	return render_template('index.html', name='Joe')

@app.route('/receiver', methods = ['POST'])
def worker():
    # read json + reply
    data = request.get_json(force=True)
    news_link = data['content']
    print (news_link)
    nlp = spacy.load('en')
    input_1 = read_input(news_link)
    train_data = read_train("TI")
    print (train_data)
    similarity = train_data.apply(lambda row: nlp(input_1).similarity(nlp(row['Article'])), axis=1)
    label = train_data['Label']
    similarity = pd.DataFrame(data={'Score': similarity, 'Label':label})
    result_avg = similarity.groupby('Label', as_index=False).mean()
    print (result_avg)
    decision = result_avg[result_avg['Score'] == result_avg['Score'].max()]
    return (decision['Label'].values[0])

def read_input(name):
    webUrl = urllib.request.urlopen(name)
    data = webUrl.read()
    soup = BeautifulSoup(data, 'lxml')
    content = soup.find('div', class_='wsw').find_all('p')
    to_pass = ""
    for i in content:
        to_pass = to_pass+str(i.string)
    return (to_pass)

def read_train(name):
    file_path=os.path.join("data", "{}.csv".format(str(name)))

    df = pd.read_csv(file_path, sep="|")
    return df


if __name__ == "__main__":
    app.run()



