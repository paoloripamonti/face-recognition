import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='facenet-face-recognition',
    version='0.2',
    author="Paolo Ripamonti",
    author_email="paolo.ripamonti93@gmail.com",
    description="MTCNN + FaceNet -> Face Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paoloripamonti/face-recognition.git",
    packages=setuptools.find_packages(),
    keywords=['facenet', 'opencv', 'deep learning', 'machine learning'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "opencv-python",
        "mtcnn",
        "sklearn",
        "tqdm",
        "numpy",
        "keras_facenet",
        "imutils",
        "tensorflow==2.4.0",
        "pandas"
    ]
)
