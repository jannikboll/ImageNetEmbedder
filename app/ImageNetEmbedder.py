import sys
sys.path.append("models/research/slim")
import nets.inception_resnet_v2 as inception_resnet_v2
#import preprocessing.inception_preprocessing as preprocess
from nets.inception_utils import inception_arg_scope, slim
import tensorflow as tf
#from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.platform import gfile
import time
#from PIL import Image
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class ImageNetEmbedder():
    
    EMBED_SIZE = inception_resnet_v2.inception_resnet_v2.default_image_size #299
    
    def __init__(self, modelPath=None, classesPath=None):
        if modelPath is None:
            self.modelPath = "data/frozen_model.pb"
        else:
            self.modelPath = modelPath
            
        if classesPath is None:
            self.classesPath = "data/imagenet_1000_classes.txt" 
        else:
            self.classesPath = classesPath
            
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
            
        self.__loadGraph(self.modelPath)
        
    def close(self):
        self.session.close()
        
    def __loadGraph(self, modelPath):
        T0 = time.time()
        with self.graph.as_default():
            with self.session.as_default():      
                with gfile.FastGFile(modelPath,'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')   
                self.input = self.graph.get_tensor_by_name('input_images:0')
                self.images = self.graph.get_tensor_by_name('processed_images:0')
                self.predictions = self.graph.get_tensor_by_name('predictions:0')
                self.embeddings = self.graph.get_tensor_by_name('embeddings:0')  
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


    def CreateFrozenGraph(checkpointFile,outputFile):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with slim.arg_scope(inception_arg_scope()):
                images_input = tf.placeholder(tf.float32,[None,None,None,3],'input_images')
                images_processed = tf.image.resize_images(images_input, (ImageNetEmbedder.EMBED_SIZE, ImageNetEmbedder.EMBED_SIZE))
                images_processed = tf.subtract(images_processed, 0.5) #Inception net preprocessing
                images_processed = tf.multiply(images_processed, 2.0)
                images_processed = tf.identity(images_processed, name="processed_images")
                _,end_points = inception_resnet_v2.inception_resnet_v2(images_processed, is_training=False, create_aux_logits=False)
                predictions = tf.identity(end_points['Predictions'], name="predictions")
                embeddings = tf.identity(end_points['PreLogitsFlatten'], name="embeddings")

                restorer = tf.train.Saver()
                restorer.restore(sess, checkpointFile)

                output_graph_def = tf.graph_util.convert_variables_to_constants(sess,graph.as_graph_def(),
                                                                                ["processed_images",
                                                                                 "predictions",
                                                                                 "embeddings"])
                with tf.gfile.GFile(outputFile, "wb") as f:
                    f.write(output_graph_def.SerializeToString())
#ImageNetEmbedder.CreateFrozenGraph("./data/inception_resnet_v2_2016_08_30.ckpt","./data/frozen_model.pb")

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
