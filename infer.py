import numpy as np
import tensorflow as tf
from scipy import ndimage

session = tf.Session()
saver = tf.train.import_meta_graph("inference/test.meta")
saver.restore(session, tf.train.latest_checkpoint("inference"))

graph = tf.get_default_graph()
dataset = graph.get_tensor_by_name("dataset:0")
model = graph.get_tensor_by_name("model:0")

pathname = "cliff_numbers.tga"
pixel_depth = 255.0
image_data = (ndimage.imread(pathname).astype(float) - pixel_depth / 2) / pixel_depth
image_data = image_data[np.newaxis, ...]

feed_dict = {dataset: image_data}
print(session.run(model, feed_dict))
