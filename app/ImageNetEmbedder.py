import sys
sys.path.append("models/research/slim")
import nets.inception_resnet_v2 as inception_resnet_v2
from nets.inception_utils import inception_arg_scope, slim
import tensorflow as tf
import numpy as np
import time

class ImageNetEmbedder():
    
    EMBED_SIZE = inception_resnet_v2.inception_resnet_v2.default_image_size #299
    
    def __init__(self, modelCheckpoint=None, classesPath=None):
        if modelCheckpoint is None:
            self.modelCheckpoint = "data/inception_resnet_v2_2016_08_30.ckpt"
        else:
            self.modelCheckpoint = modelCheckpoint
            
        if classesPath is None:
            self.classesPath = "data/imagenet_1000_classes.txt" 
        else:
            self.classesPath = classesPath
            
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
            
        self.__loadGraph(self.modelCheckpoint)
        
    def close(self):
        self.session.close()
        
    def __loadGraph(self, modelCheckpoint):
        T0 = time.time()
        with self.graph.as_default():
            with self.session.as_default():
                with slim.arg_scope(inception_arg_scope()): 

                    #Create hooks into model and a nicer input interface
                    self.input = tf.placeholder(tf.float32,[None,None,None,3],'input_images')
                    images_processed = tf.image.resize_images(self.input, (ImageNetEmbedder.EMBED_SIZE, ImageNetEmbedder.EMBED_SIZE))
                    images_processed = tf.subtract(images_processed, 0.5) #Inception net preprocessing
                    images_processed = tf.multiply(images_processed, 2.0) 
                    self.images = tf.identity(images_processed, name="processed_images")
                    _,end_points = inception_resnet_v2.inception_resnet_v2(images_processed, is_training=False, create_aux_logits=False)
                    self.predictions = tf.identity(end_points['Predictions'], name="predictions")
                    self.embeddings = tf.identity(end_points['PreLogitsFlatten'], name="embeddings")
                    
                    #Restore weights from checkpoint
                    restorer = tf.train.Saver()
                    restorer.restore(self.session, modelCheckpoint)

        T1 = time.time()
        print("Time loading model: %2.2fs"%(T1-T0))
        
        #Load class labels
        with open(self.classesPath) as f:
            classes = f.readlines()
        self.classes = [x.strip() for x in classes] 
        self.classes.insert(0,"Unknown")
        
        
    def embedAndLabel(self, images, cutoffProb=0.05):
        with self.graph.as_default():
            with self.session.as_default():
                T0 = time.time()
                (predictions_eval,embedding_eval) = self.session.run([self.predictions,self.embeddings], {self.input:images})
                T1 = time.time()
                print("Time embedding %d images: %2.2fs (%2.2fs/im)"%(len(images),T1-T0,(T1-T0)/len(images)))
                labels = []
                for prediction in predictions_eval:
                    sortIDs = np.argsort(prediction)[::-1]
                    lblIDs = np.where(prediction[sortIDs] > cutoffProb)[0]
                    labels.append([(str(np.array(self.classes)[sortIDs][lblID]),float(prediction[sortIDs][lblID])) for lblID in lblIDs])
                return labels, embedding_eval
                    
            
    def getNetProcessedImages(self, images):
        with self.graph.as_default():
            with self.session.as_default():
                T0 = time.time()
                images_eval = self.session.run(self.images, {self.input:images})
                T1 = time.time()
                print("Time embedding %d images: %2.2fs (%2.2fs/im)"%(len(images),T1-T0,(T1-T0)/len(images)))
                return images_eval
     


#from matplotlib import pyplot as plt
#from PIL import Image
#embedder = ImageNetEmbedder()
#
#images = ["./data/images/Cat.jpg",]
#for imName in images:
#    im = np.array(Image.open(imName))/255.0
#    plt.figure()
#    plt.imshow(im)
#    plt.show()
#    labels,embeddings = embedder.embedAndLabel([im,])
#    pIms = embedder.getNetProcessedImages([im,])
#    plt.figure()
#    plt.imshow(pIms[0])
#    plt.show()
#    for label in labels[0]:
#        print("%2.2f \t %s"%(label[1],label[0]))
