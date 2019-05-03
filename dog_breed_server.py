
import os
from sklearn.externals import joblib
from flask import Flask, request, redirect, flash, jsonify
from werkzeug.utils import secure_filename

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pickle

UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def make_prediction(filename, top=5):
    status = 'SUCCESS'
    
    # ResNEt50 prediction    
    img = image.load_img(filename, target_size=(224, 224))
    X = np.expand_dims(image.img_to_array(img), axis=0)
    X = preprocess_input(X)

    model = ResNet50(weights='imagenet')
    preds = model.predict(X)
    y_pred = decode_predictions(preds, top=top)[0]
    y_pred = { x[1]: "{:.5f}".format(float(x[2])) for x in y_pred}
    
    # Own model prediction
    infile = open('VGG16_224x224x3_D512_D128.model','rb')
    model = pickle.load(infile)
    infile.close()

    infile = open('class_indices','rb')
    class_indices = pickle.load(infile)
    infile.close()

    img = image.load_img(filename, target_size=(224, 224))
    tmp = np.expand_dims(img, axis=0)       
    tmp2 = tmp / 255
    
    own_pred = model.predict_proba(tmp2)
    y_own_pred = {}
    for breed, pred in zip(list(class_indices.keys()),own_pred[0]):
        y_own_pred[breed] = "{:.5f}".format(pred)
            
    return status, y_pred, y_own_pred

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_path)
                    
            status, ResNet50Pred, VGG16_own_model = make_prediction(full_path)
            
            # usuwam zdfdjęcie z serwera po predykcji
            os.remove(full_path)

            return jsonify({
                'filename': filename,
                'ResNet50Pred': ResNet50Pred,
                'VGG16_own_model': VGG16_own_model,
                'status': status
            })
        
    return '''
    <!doctype html>
    <title>Klasyfikacja zdjęć</title>
    <h1>Klasyfikacja zdjęć</h1>
    <p>Wybierz zdjęcie, następnie klikniaj "upload"</p>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
