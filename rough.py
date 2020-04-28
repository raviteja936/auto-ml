import tensorflow as tf
import os

IMG_WIDTH = 224
IMG_HEIGHT = 224
CLASS_NAMES = ['daisy', 'roses', 'dandelion', 'sunflowers', 'tulips']
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

file_path =  b'/home/ravi/.keras/datasets/flower_photos/tulips/5043225469_0aa23f3c8f_n.jpg'
img = tf.io.read_file(file_path)
print(decode_img(img))