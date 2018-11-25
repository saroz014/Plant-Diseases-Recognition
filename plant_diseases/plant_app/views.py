from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K

# Create your views here.

out_dict = {'Apple___Apple_scab': 0,
            'Apple___Black_rot': 1,
            'Apple___Cedar_apple_rust': 2,
            'Apple___healthy': 3,
            'Blueberry___healthy': 4,
            'Cherry_(including_sour)___Powdery_mildew': 5,
            'Cherry_(including_sour)___healthy': 6,
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7,
            'Corn_(maize)___Common_rust_': 8,
            'Corn_(maize)___Northern_Leaf_Blight': 9,
            'Corn_(maize)___healthy': 10,
            'Grape___Black_rot': 11,
            'Grape___Esca_(Black_Measles)': 12,
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
            'Grape___healthy': 14,
            'Orange___Haunglongbing_(Citrus_greening)': 15,
            'Peach___Bacterial_spot': 16,
            'Peach___healthy': 17,
            'Pepper,_bell___Bacterial_spot': 18,
            'Pepper,_bell___healthy': 19,
            'Potato___Early_blight': 20,
            'Potato___Late_blight': 21,
            'Potato___healthy': 22,
            'Raspberry___healthy': 23,
            'Soybean___healthy': 24,
            'Squash___Powdery_mildew': 25,
            'Strawberry___Leaf_scorch': 26,
            'Strawberry___healthy': 27,
            'Tomato___Bacterial_spot': 28,
            'Tomato___Early_blight': 29,
            'Tomato___Late_blight': 30,
            'Tomato___Leaf_Mold': 31,
            'Tomato___Septoria_leaf_spot': 32,
            'Tomato___Spider_mites Two-spotted_spider_mite': 33,
            'Tomato___Target_Spot': 34,
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35,
            'Tomato___Tomato_mosaic_virus': 36,
            'Tomato___healthy': 37}

li = list(out_dict.keys())

def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        m = str(filename)
        K.clear_session()
        im = Image.open("media/" + m)
        j = im.resize((224, 224),)
        l = "predicted.jpg"
        j.save("media/" + l)
        file_url = fs.url(l)
        mod = load_model('plant_app/model.hdf5', compile=False)
        img = image.load_img(myfile, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        preds = mod.predict(x)
        d = preds.flatten()
        j = d.max()
        for index, item in enumerate(d):
            if item == j:
                result = li[index]
                return render(request, "plant_app/index.html", {
                                 'result': result, 'file_url': file_url })

    return render(request, "plant_app/index.html")
