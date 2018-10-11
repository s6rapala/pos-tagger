from re import split
import pickle
import os
from nltk.corpus import ConllCorpusReader
from flask import (Request, Flask, json, request, flash,
                   redirect, render_template)
from werkzeug.utils import secure_filename

from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

app = Flask(__name__)

UPLOAD_FOLDER = '/home/raghu/Downloads/pos-tagger/templates'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload')
def upload_file_html():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['corpusFile']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'File Uploaded Successfully'


@app.route('/tagger/api/v1.0/ingest', methods=['GET', 'POST'])
def ingest_train_data():
    global tagger
    # corpusFile = "tiger_release_aug07.corrected.16012013.conll09" #request.args['corpusFile']"
    taggerFile = 'SerializedTagger.pickle'
    corp = ConllCorpusReader('../templates', 'tiger_release_aug07.corrected.16012013.conll09',
                                         ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                         encoding='utf-8')
    tagged_sents = corp.tagged_sents()
    split_percentage = 0.25
    split_size = int(len(tagged_sents) * split_percentage)
    train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]
    tagger = ClassifierBasedGermanTagger(train=train_sents)
    with open(taggerFile, 'wb') as output:
        pickle.dump(tagger, output)
    return json.dumps({'message': 'data ingested'})


@app.route('/tagger/api/v1.0/pos', methods=['GET', 'PUT'])
def api_message():
    with open('SerializedTagger.pickle', 'rb') as f:
        tagger = pickle.load(f)

    res_items = tagger.tag(['Das', 'ist', 'ein', 'einfacher', 'Test'])

    pos = {}
    for item in res_items:
        pos[item[0]] = item[1]
    return json.dumps(pos, sort_keys=False, indent=4)

if __name__=='__main__':
    app.run(debug=True)