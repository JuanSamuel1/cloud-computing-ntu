from flask import Flask, request, make_response, jsonify
from http import HTTPStatus
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import math

class TextSearcher:
    def __init__(self):
        self.id_counter = 1
        self.text_ids = []
        self.text_storage = []
        self.available_keywords = []

def TextRank(text):
    # Copied from Jupyter Notebook
    def clean(text):
        text = text.lower()
        printable = set(string.printable)
        text = filter(lambda x: x in printable, text)
        text = "".join(list(text))
        return text
    
    cleaned_text = clean(text)
    text = word_tokenize(cleaned_text)
    POS_tag = nltk.pos_tag(text)
    
    wordnet_lemmatizer = WordNetLemmatizer()

    adjective_tags = ['JJ','JJR','JJS']

    lemmatized_text = []

    for word in POS_tag:
        if word[1] in adjective_tags:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a")))
        else:
            lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun

    POS_tag = nltk.pos_tag(lemmatized_text)
    stopwords = []

    wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 

    for word in POS_tag:
        if word[1] not in wanted_POS:
            stopwords.append(word[0])

    punctuations = list(str(string.punctuation))

    stopwords = stopwords + punctuations

    stopword_file = open("long_stopwords.txt", "r")

    lots_of_stopwords = []

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    stopwords_plus = []
    stopwords_plus = stopwords + lots_of_stopwords
    stopwords_plus = set(stopwords_plus)

    processed_text = []
    for word in lemmatized_text:
        if word not in stopwords_plus:
            processed_text.append(word)
            
    vocabulary = list(set(processed_text))
    vocab_len = len(vocabulary)

    weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

    score = np.zeros((vocab_len),dtype=np.float32)
    window_size = 3
    covered_coocurrences = []

    for i in range(0,vocab_len):
        score[i]= 1/vocab_len
        for j in range(0,vocab_len):
            if j==i:
                weighted_edge[i][j]=1
            else:
                for window_start in range(0,(len(processed_text)-window_size)):
                    
                    window_end = window_start+window_size
                    
                    window = processed_text[window_start:window_end]
                    
                    if (vocabulary[i] in window) and (vocabulary[j] in window):
                        
                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])
                        
                        if [index_of_i,index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                            covered_coocurrences.append([index_of_i,index_of_j])
    
    weighted_edge /= weighted_edge.sum(axis=0, keepdims=True)
    MAX_ITERATIONS = 50
    d=0.85
    threshold = 0.0001 

    for iter in range(0,MAX_ITERATIONS):
        new_score = weighted_edge.dot(score)
        new_score = ((1-d) / vocab_len) + (d * new_score)
        if np.sum(np.fabs(new_score-score)) <= threshold: 
            print("Converging at iteration "+str(iter)+"....")
            break
        score = new_score
    
    pairs = [(vocabulary[i], score[i]) for i in range(vocab_len)]
    pairs.sort(key= lambda x: x[1], reverse=True)

    res = []
    for t in range(5):
        res.append(pairs[t][0])

    phrases = []

    phrase = " "
    for word in lemmatized_text:
        
        if word in stopwords_plus:
            if phrase!= " ":
                phrases.append(str(phrase).strip().split())
            phrase = " "
        elif word not in stopwords_plus:
            phrase+=str(word)
            phrase+=" "
    
    unique_phrases = []

    for phrase in phrases:
        if phrase not in unique_phrases:
            unique_phrases.append(phrase)
    
    for word in vocabulary:
        for phrase in unique_phrases:
            if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
                unique_phrases.remove([word])
    
    phrase_scores = []
    keywords = []
    for phrase in unique_phrases:
        phrase_score=0
        keyword = ''
        for word in phrase:
            keyword += str(word)
            keyword += " "
            phrase_score+=score[vocabulary.index(word)]
        phrase_scores.append(phrase_score)
        keywords.append(keyword.strip())
    

    sorted_index = np.flip(np.argsort(phrase_scores),0)

    for i in range(5):
        res.append(keywords[sorted_index[i]])

    return res

textSearcher = TextSearcher()

# Create the Flask application instance
app = Flask(__name__)

# Set current_app context
app.app_context().push()


@app.route('/text', methods=['GET'])
def get_text():
    keyword =  request.args.get('keyword')

    if not keyword:
        return make_response(jsonify(
                {"message": "invalid keyword"}
            ), HTTPStatus.BAD_REQUEST)
    
    res = []
    for i in range(len(textSearcher.available_keywords)):
        if keyword in textSearcher.available_keywords[i]:
            res.append(textSearcher.text_storage[i])
    

    return make_response(jsonify(
                {"message": "OK",
                 "result_text": res}
            ), HTTPStatus.OK)

@app.route('/text', methods=['POST'])
def post_text():
    text = request.json["text"]

    if not text:
        return make_response(jsonify(
                {"message": "invalid text"}
            ), HTTPStatus.BAD_REQUEST)
    
    res = TextRank(text)
    textSearcher.text_ids.append(textSearcher.id_counter)
    textSearcher.id_counter += 1
    textSearcher.text_storage.append(text)
    textSearcher.available_keywords.append(res)
    print(res)
    return make_response(jsonify(
                {"message": "OK"}
            ), HTTPStatus.OK)

@app.route('/text', methods=['PUT'])
def update_text():
    id = request.json["id"]
    text = request.json["text"]

    if not text:
        return make_response(jsonify(
                {"message": "invalid text"}
            ), HTTPStatus.BAD_REQUEST)
    
    if not id or id not in textSearcher.text_ids:
        return make_response(jsonify(
                {"message": "invalid id"}
            ), HTTPStatus.BAD_REQUEST)
    
    idx = textSearcher.text_ids.index(id)
    res = TextRank(text)
    textSearcher.text_storage[idx] = text
    textSearcher.available_keywords[idx] = res
    return make_response(jsonify(
                {"message": "OK"}
            ), HTTPStatus.OK)

@app.route('/text', methods=['DELETE'])
def delete_text():
    id = request.json["id"]
    
    if not id or id not in textSearcher.text_ids:
        return make_response(jsonify(
                {"message": "invalid id"}
            ), HTTPStatus.BAD_REQUEST)
    
    idx = textSearcher.text_ids.index(id)
    textSearcher.available_keywords.pop(idx)
    textSearcher.text_storage.pop(idx)
    textSearcher.text_ids.pop(idx)

    return make_response(jsonify(
                {"message": "OK"}
            ), HTTPStatus.OK)

@app.route('/metadata', methods=['PATCH'])
def update_metadata():
    id = request.json["id"]
    keywords = request.json["keywords"]

    if not id or id not in textSearcher.text_ids:
        return make_response(jsonify(
                {"message": "invalid id"}
            ), HTTPStatus.BAD_REQUEST)
    
    k = keywords.split(",")
    idx = textSearcher.text_ids.index(id)
    textSearcher.available_keywords[idx] = k
    
    return make_response(jsonify(
                {"message": "OK"}
            ), HTTPStatus.OK)



app.run('0.0.0.0','8080')