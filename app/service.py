import io
import traceback
import numpy as np
import time
from PIL import Image
from flask import Flask, request, jsonify
from ImageNetEmbedder import ImageNetEmbedder

app = Flask(__name__) #create the Flask app

#Create Embed service
Embedder = ImageNetEmbedder()

def processImageUpload():
    """
    Image upload processing.
    """

    #Define return dictionary
    retDict = {}
    retDict['model'] = Embedder.modelPath.split("/")[-1]
    
    #Read data
    f = request.files['image']
    if(str(f.filename) is not ''):
        retDict['filename'] = f.filename
        #Store data in memory
        in_memory_file = io.BytesIO()
        f.save(in_memory_file)
        #Convert to image
        image = Image.open(in_memory_file)
        image = np.array(image)
    else:
        raise Exception("Image field empty.")
        
    #Prepare image:
    if(image.dtype == np.uint8):
        image = image/255.0
        
    #Do embedding
    T0 = time.time()
    labels, embeddings = Embedder.embedAndLabel([image,])
    T1 = time.time()
    retDict['time'] = T1-T0
    retDict['labels'] = labels[0] #We only embed 1 object
    retDict['embedding'] = embeddings[0].tolist()      
    return retDict


#Root site
@app.route('/', methods=['GET', 'POST']) #allow both GET and POST requests
def default():
    """
    Default entrypoint for webservice.
    """
    reponseDict = {}
    try:
        #Are we processing an input?
        if request.method == 'POST' and 'image' in request.files:  #Is it an image?
            reponseDict.update(processImageUpload())
        #If not processing, show form:
        else:
            reponseDict["message"] = "Please post to field 'image' with an image file."
    except Exception as e:
        reponseDict["message"] = "An error occurred: %s\n"%e
        reponseDict["message"] += traceback.format_exc()   
    
    return jsonify(reponseDict)

#Start webservice
if __name__ == '__main__':
    app.run(debug=False, port=8888)
