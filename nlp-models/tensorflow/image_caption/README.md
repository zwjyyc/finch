* The data downloading, processing, and loading is based on [yunjie's awesome work](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

* We implement the model in TensorFlow instead of PyTorch

* In PyTorch, popular networks (e.g. ResNet) can be easily obtained via ```torchvision.models```; In TensorFlow, we use ```tf.contrib.keras.applications``` to load pre-trained networks, though there are less models to choose;

#### 1. Install COCO API
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
```

#### 2. Download the dataset

```bash
$ pip install -r requirements.txt
$ chmod +x download.sh
$ ./download.sh
```

#### 3. Preprocessing

```bash
$ python build_vocab.py   
$ python resize.py
```
