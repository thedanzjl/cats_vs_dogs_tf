
from train import *
import os
import cv2
import sys
from create_dataset import _bytes_feature

model = tf.estimator.Estimator(model_dir='./dogs_vs_cats-model/',
                               model_fn=model_fn)


def predict_input_fn(addr):
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feature = {'image_raw': _bytes_feature(img.tostring())}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    written = example.SerializeToString()
    keys_to_features = {"image_raw": tf.FixedLenFeature([], tf.string)}
    parsed = tf.parse_single_example(written, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    return {'image': image}


def predict(filename):
    labels = ['cats', 'dogs']
    result = next(model.predict(input_fn=lambda: predict_input_fn(filename)))
    return labels[result]


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == '--dataset':
        images = os.listdir('dataset/test_set')
        for image in images:
            image = os.path.join('dataset/test_set', image)
            read_image = cv2.imread(image)
            result = predict(image)
            cv2.imshow(result, read_image)
            cv2.waitKey()
    elif sys.argv[1] == '--image' and sys.argv[2]:
        image = sys.argv[2]
        read_image = cv2.imread(image)
        result = predict(image)
        print(result)


