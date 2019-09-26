# Face Recognition

Simple library to recognize faces from given images

[<img src="https://www.kaggleusercontent.com/kf/21132673/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..Hig7ClqLMXecFinIUQqE_g.Fg1LGL5rc0-PLv7twHxDXzUI8oK3nZGyfpUDv_P-le31C2I2-qoFjWzaxz1n1WlhJqaFyzkqMlfptZy65zaCvk5bxQ5h4bEeS6AFTcORFUdYfMqBM5cYRvWKg4KX-sM-SeWRPRh_s-HWIxe4m2gZWw.l8bObQ1Fhvykp8XMeIAbXw/__results___files/__results___23_0.png">](https://www.kaggle.com/paoloripamonti/face-recogniton)

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

[<img src="https://www.kaggleusercontent.com/kf/21129215/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..HbixuFgKpuPhmZNn6lkvkQ.ll6VOKAnA0aeJQpm-v9L0hYYZzIAfvvTa1TAxVzJP-bRDPwEpH1WYyrgrX4Vr_ADSI929jwLSGSuuq9KGJnQogJbYVPTRkGW5pBxO8R4rhxrSkg8IoQ6pokYR6ZtehZvjKbK01Bjkow6ykbFWZmZcA.xUV-JT8XBPEHdZJm-yasZQ/__results___files/__results___20_0.png">](https://www.kaggle.com/paoloripamonti/face-recogniton)

[<img src="https://www.kaggleusercontent.com/kf/21129215/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..HbixuFgKpuPhmZNn6lkvkQ.ll6VOKAnA0aeJQpm-v9L0hYYZzIAfvvTa1TAxVzJP-bRDPwEpH1WYyrgrX4Vr_ADSI929jwLSGSuuq9KGJnQogJbYVPTRkGW5pBxO8R4rhxrSkg8IoQ6pokYR6ZtehZvjKbK01Bjkow6ykbFWZmZcA.xUV-JT8XBPEHdZJm-yasZQ/__results___files/__results___21_0.png">](https://www.kaggle.com/paoloripamonti/face-recogniton)


For more details you can see: [https://www.kaggle.com/paoloripamonti/face-recogniton](https://www.kaggle.com/paoloripamonti/face-recogniton)
