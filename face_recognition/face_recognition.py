import base64
import glob
import os
import time
import warnings

import cv2
import imutils
import math
import numpy as np
import tensorflow as tf
import tqdm
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN

from face_recognition import config
from face_recognition.euclidean_classifier import EuclideanClassifier

warnings.filterwarnings("ignore")


class FaceRecognition(object):
    """
    Face Recognition object class
    """

    def __init__(self):
        """
        Initialize Face Recognition model
        """
        # Load Face Detector
        self.face_detector = MTCNN()

        # Load FaceNet
        self.facenet = FaceNet()

        # Euclidean Classifier
        self.clf = None

    def predict(self, path, threshold=None):
        """
        Find faces and recognize them, return predicted people into image
        :param path: Source image path
        :param threshold: cutoff threshold
        :return: Return predictions and images with rectangles drawn
        """
        if not self.clf:
            raise RuntimeError("No classifier found. Please load classifier")

        start_at = time.time()
        bounding_boxes = []
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.uint8)
        for person, confidence, box in self.__predict__(image, threshold=threshold):
            # Draw rectangle with person name
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, person, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

            bounding_boxes.append({
                "person": person,
                "confidence": confidence,
                "box": box,
            })

        # encode frame
        _, buffer = cv2.imencode('.jpg', image)

        return {
            "frame": base64.b64encode(buffer).decode('ascii'),
            "elapsed_time": (time.time() - start_at),
            "predictions": bounding_boxes

        }

    def __predict__(self, image, threshold=None):
        """
        Extract face and perform evaluation
        :param image: Source image
        :param threshold: decision threshold
        :return:  yield (person_id, person, confidence, box)
        """
        # Resize Image
        for encoding, face, box in self.face_encoding(image):
            # Check face size
            if (box[2] - box[0]) < config.MIN_FACE_SIZE[0] or \
                    (box[3] - box[1]) < config.MIN_FACE_SIZE[1]:
                yield (config.UNKNOWN_LABEL, 0.0, box)
            else:
                results = self.clf.predict(encoding)
                person, confidence = results["person"], results["confidence"]
                if threshold and confidence < threshold:
                    person = config.UNKNOWN_LABEL

                yield (person, confidence, box)

    def face_detection(self, image):
        """
        Face detection from source image
        :param image: Source image
        :return: extracted face and bounding box
        """
        image_to_detect = image.copy()

        # detect faces in the image
        for face_attributes in self.face_detector.detect_faces(image_to_detect):
            if face_attributes["confidence"] > config.FACE_CONFIDENCE:
                # extract the bounding box
                x1, y1, w, h = [max(point, 0) for point in face_attributes["box"]]
                x2, y2 = x1 + w, y1 + h

                face = image[y1:y2, x1:x2]
                # Align face
                face = FaceRecognition.align_face(face_attributes, face.copy())

                yield (cv2.resize(face, config.FACE_SIZE), (x1, y1, x2, y2))

    def face_encoding(self, source_image):
        """
        Extract face encodings from image
        :param source_image: Source image
        :return: 512 encoding, face and bounding box
        """
        for face, box in self.face_detection(source_image):
            # Face encoding
            encoding = self.facenet.embeddings(np.expand_dims(face, axis=0))[0]

            yield encoding, face, box

    @staticmethod
    def align_face(face_attribute, image):
        if not face_attribute:
            return image
        # Get left and right eyes
        left_eye = face_attribute["keypoints"]["left_eye"]
        right_eye = face_attribute["keypoints"]["right_eye"]
        # Get distance between eyes
        d = math.sqrt(math.pow(right_eye[0] - left_eye[0], 2) + math.pow(right_eye[1] - left_eye[1], 2))
        a = left_eye[1] - right_eye[1]
        # get alpha degree
        alpha = (math.asin(a / d) * 180.0) / math.pi

        return imutils.rotate(image, -alpha)

    def load(self, path):
        """
        Load classifier from pickle file
        :param path: path
        """
        clf = EuclideanClassifier()
        clf.load(path)

        self.clf = clf

    def save(self, path):
        """
        Save classifier as pickle file
        :param path: path
        """
        self.clf.save(path)

    def fit(self, folder):
        """
        Fit classifier from directory.
        Directory must have this structure:
            Person 1:
                file.jpg
                ....
                file.jpg
            Person 2:
                file.jpg
                ...
                file.jpg
            ...
        :param folder: root folder path
        """
        # Initialize classifier
        clf = EuclideanClassifier()

        # Load all files
        files = []
        for ext in config.ALLOWED_IMAGE_TYPES:
            files.extend(glob.glob(os.path.join(folder, "*", ext), recursive=True))

        for path in tqdm.tqdm(files):
            # Load image
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get person name by folder
            person = os.path.split(os.path.split(path)[0])[1]

            # Get encoding
            for encoding, face, box in self.face_encoding(image):
                # Add to classifier
                clf.fit([encoding], [person])

        self.clf = clf

    def fit_from_dataframe(self, df, person_col="person", path_col="path"):
        """
        Fit classifier from dataframe.
        :param df: Pandas dataframe
        :param person_col: Dataframe column with person id
        :param path_col: Dataframe column with image path
        """
        # Initialize classifier
        clf = EuclideanClassifier()

        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            # Load image
            image = cv2.imread(row[path_col])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get person name by folder
            person = row[person_col]

            # Get encoding
            for encoding, face, box in self.face_encoding(image):
                # Add to classifier
                clf.fit([encoding], [person])

        self.clf = clf
