# ImageNet Embedder Webservice

This project exposes a pre-trained Tensorflow ImageNet model as a webservice. It takes an image as input and returns ImageNet labels and embedding. The project uses the Inception-ResNet-v2 implementation from the [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim) model library.

## Requirements

* Python 3
* [Tensorflow](https://www.tensorflow.org/install/)
* Flask (pip install flask)
* Pretrained Inception-ResNet-v2 model (.ckpt). Can be found in the Tensorflow-Slim pretrained section [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

Recommended is also [Docker](https://store.docker.com/search?type=edition&offering=community) to pack your service in an easily sharable container.

## Running the service from Python 3
Simply run the service.py script. This boots a non-production webservice on localhost:8888.
```
python service.py
```

## Building & running the service using Docker
Build the docker image once, using either the .gpu og .cpu file (Depending on your hardware).
After a successful build, Docker can be called to run the image, exposing the service on localhost:8888.
```
docker build -f Dockerfile.cpu -t imagenetembedder .
docker run -it -p 8888:8888 imagenetembedder
```

## Using the webservice
To use the service, make a POST request with multipart/form-data, assigning an image to the `image`-field.
```
curl localhost:8888 -F image=@myimage.jpg
```
This returns:
```json
{
    "filename":"myimage.jpg",
    "labels":[["tabby, tabby cat",0.58],["tiger cat",0.21],["Egyptian cat",0.11]],
    "model":"inception_resnet_v2_2016_08_30.ckpt",
    "time":1.1794331073760986
}
```
