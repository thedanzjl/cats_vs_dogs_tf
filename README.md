# Image classification with tensorflow


<b>Dataset</b> is taken [here](https://www.kaggle.com/siddarthareddyt/cats-and-dogs)



1) Install dependencies

        pip install tensorflow opencv2-python numpy

2) Generate csv files with image paths and correspoding labels, runnig generate_csv.py

        python generate_csv.py

3) Create training.tfrecords and validation.tfrecords files for tensorflow Dataset API via

        python create_dataset.py
    
4) Train and evaluate using

        python train.py
    
5) Use "predict-playground.py" to make predictions for test set or for your own image

        python predict-playground.py --dataset 
        python predict-playground.py --image path/to/image.jpg
    
