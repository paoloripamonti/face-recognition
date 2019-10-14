# Face Recognition

Simple library to recognize faces from given images

[<img src="https://raw.githubusercontent.com/paoloripamonti/face-recognition/master/img/home.png">](https://www.kaggle.com/paoloripamonti/face-recogniton)

### Face Recognition pipeline

Below the pipeline for face recognition:
- **Face Detection**: the [MTCNN](https://github.com/ipazc/mtcnn) algorithm is used to do face detection
- **Face Alignement** Align face by eyes line
- **Face Encoding** Extract encoding from face using [FaceNet](https://github.com/faustomorales/keras-facenet)
- **Face Classification** Classify face via eculidean distrances between face encodings

### How to install
```git
pip install git+https://github.com/paoloripamonti/face-recognition
```

### How to train custom model

Initialize model
```python
from face_recognition import FaceRecognition

fr = FaceRecognition()
```

#### Train model with pandas DataFrame:

```python
fr = FaceRecognition()

fr.fit_from_dataframe(df)
```

where 'df' is pandas DataFrame with column **person** (person name) and column **path** (image path)

#### Train model with folder:

```python
fr = FaceRecognition()

fr.fit('/path/root/')
```

the root folder must have the following structure:

```
root\
    Person_1\
        image.jpg
        ...
        image.jpg
    Person_2\
        image.jpg
        ...
        image.jpg
    ...
        
```

### Save and load model

you can save and load model as **pickle** file.


```python
fr.save('model.pkl')
```

```python
fr = FaceRecognition()

fr.load('model.pkl')
```


### Predict image

```python
fr.predict('/path/image.jpg')
```

Recognize faces from given image.
The output is a JSON with folling structure:

```
{
  "frame": "image data", # base64 image with bounding boxes
  "elapsed_time": time, # elapsed time in seconds
  "predictions": [
      {
        "person": "Person", # person name
        "confidence": float, # prediction confidence
        "box": (x1, y1, x2, y2) # face bounding box
      }
  ]
}
```

### Example

[<img src="https://raw.githubusercontent.com/paoloripamonti/face-recognition/master/img/test1.png">](https://www.kaggle.com/paoloripamonti/face-recogniton)

[<img src="https://raw.githubusercontent.com/paoloripamonti/face-recognition/master/img/test2.png">](https://www.kaggle.com/paoloripamonti/face-recogniton)


For more details you can see: [https://www.kaggle.com/paoloripamonti/face-recogniton](https://www.kaggle.com/paoloripamonti/face-recogniton)
