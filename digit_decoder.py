from __future__ import print_function
import os
import sys
import tarfile
import pickle
from six.moves.urllib.request import urlretrieve
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import digitStruct

last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(url, expected_bytes):
    filename = os.path.basename(url)
    print("attempting to download {}...".format(filename))
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    return root


def load_digit_data(data_file):
    print("getting digit data from .mat file: {}".format(data_file))
    # get the digit bounding boxes
    data = {}
    for ds_obj in digitStruct.yieldNextDigitStruct(data_file):
        digits = []
        for bbox in ds_obj.bboxList:
            digits.append({"digit": bbox.label, "box":{
                "left": bbox.left,
                "top": bbox.top,
                "width": bbox.width,
                "height": bbox.height}
            })
        data[ds_obj.name] = digits
        # if len(data.keys()) >= 2000:  # temporary, to keep fast
        #     break

    return data


def get_uber_bounding_box(digit_data):
    """
    Get the larger bounding box from the digit data (from the individual bounding boxes)
    """

    '''
    print("getting uber box for:")
    for d in digit_data:
        print("digit {}: {}, {}, {}, {}".format(
            d["digit"],
            d["box"]["left"],
            d["box"]["top"],
            d["box"]["left"] + d["box"]["width"],
            d["box"]["top"] + d["box"]["height"]
        ))
    '''

    left_most = min([d["box"]["left"] for d in digit_data])
    top_most = min([d["box"]["top"] for d in digit_data])
    right_most = max([d["box"]["left"] + d["box"]["width"] for d in digit_data])
    bottom_most = max([d["box"]["top"] + d["box"]["height"] for d in digit_data])

    '''
    print("got: {}, {}, {}, {}".format(
        left_most, top_most, right_most, bottom_most
    ))
    '''

    uber_bbox = {"left": left_most, "top": top_most, "right": right_most, "bottom": bottom_most}

    # expand by 30%
    width = uber_bbox["right"] - uber_bbox["left"]
    height = uber_bbox["bottom"] - uber_bbox["top"]
    new_width = 1.3 * width
    new_height = 1.3 * height
    delta_width = 0.5 * (new_width - width)
    delta_height = 0.5 * (new_height - height)
    expanded_uber_bbox = {
        "left": max(0, int(uber_bbox["left"] - delta_width)),
        "right": max(0, int(uber_bbox["right"] + delta_width)),
        "top": max(0, int(uber_bbox["top"] - delta_height)),
        "bottom": max(0, int(uber_bbox["bottom"] + delta_height))
    }

    return expanded_uber_bbox


def load_images(image_folder, digit_data):
    num_samples = len(digit_data)
    images = np.ndarray(dtype=np.float32, shape=(num_samples, 64, 64, 3))
    num_digits_labels = np.ndarray(dtype=np.float32, shape=(num_samples, 5))  # labels, hot 1 encoding
    image_index = 0
    for filename, data in digit_data.items():
        pathname = os.path.join(image_folder, filename)
        if not os.path.exists(pathname):
            print("Cannot find file: {}, skipping".format(filename))
            continue
        print("loading image: {}".format(pathname))
        image_data = ndimage.imread(pathname)

        # find uber bounding box
        uber_bbox = get_uber_bounding_box(data)

        # crop
        cropped_data = image_data[
                       uber_bbox["top"]:uber_bbox["bottom"],
                       uber_bbox["left"]:uber_bbox["right"],
                       :]
        # resize to 64x64
        resized_data = ndimage.zoom(cropped_data, (64.0/cropped_data.shape[0], 64.0/cropped_data.shape[1], 1))

        # randomly crop to 54x54
        # r = np.random.randint(0, 9, 2)
        # crop2_data = resized_data[r[0]:r[0]+54, r[1]:r[1]+54]
        # images[image_index, :, :, :] = crop2_data

        images[image_index, :, :, :] = resized_data

        label = np.zeros(shape=5)
        label[len(data)-1 if len(data) < 5 else 4] = 1.0
        num_digits_labels[image_index, :] = label


        image_index += 1

    print("num_digits_labels = {}".format(num_digits_labels))
    return images, num_digits_labels


def get_data_from_pickle():
    if not os.path.exists(pickle_filename):
        return None
    with open(pickle_filename, 'rb') as f:
        result = pickle.load(f)
    return result


def pickle_data(dataset):
    print('Pickling %s.' % pickle_filename)
    try:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_filename, ':', e)

pickle_filename = 'dataset.pickle'

train_archive = maybe_download('http://ufldl.stanford.edu/housenumbers/train.tar.gz', 404141560)
test_archive = maybe_download('http://ufldl.stanford.edu/housenumbers/test.tar.gz', 276555967)
force_rebuild_data = False

# extract and split the test data into test and validation
train_folder = maybe_extract(train_archive)
test_folder = maybe_extract(test_archive)
valid_folder = "validation"
if not os.path.exists(valid_folder):
    os.mkdir(valid_folder)
    png_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if os.path.splitext(f)[1] == ".png"]
    for f in png_files[:10000]:
        os.rename(f, f.replace(test_folder, valid_folder))

dataset = get_data_from_pickle()
if force_rebuild_data or not dataset:
    print("No pickle data...")
    train_data = load_digit_data('train/digitStruct.mat')
    test_data = load_digit_data('test/digitStruct.mat')

    train_images, train_num_labels = load_images(train_folder, train_data)
    valid_images, valid_num_labels = load_images(valid_folder, test_data)  # the data is in test_data for both
    test_images, test_num_labels = load_images(test_folder, test_data)

    dataset = {
        'train_images': train_images,
        'train_num_labels': train_num_labels,
        'valid_images': valid_images,
        'valid_num_labels': valid_num_labels,
        'test_images': test_images,
        'test_num_labels': test_num_labels
    }
    print("Picking data...")
    pickle_data(dataset)
else:
    print("Found pickle data...")
    train_images = dataset['train_images']
    train_num_labels = dataset['train_num_labels']
    valid_images = dataset['valid_images']
    valid_num_labels = dataset['valid_num_labels']
    test_images = dataset['test_images']
    test_num_labels = dataset['test_num_labels']
    print("Training images: {}".format(len(train_images)))
    print("Training labels: {}".format(len(train_num_labels)))
    print("Validation images: {}".format(len(test_images)))
    print("Validation labels: {}".format(len(test_num_labels)))
    print("Test images: {}".format(len(test_images)))
    print("Test labels: {}".format(len(test_num_labels)))


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


image_size = 64
num_labels = 5
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_channels = 3

graph = tf.Graph()
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_images)
    tf_test_dataset = tf.constant(test_images)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv2 = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        shape = hidden2.get_shape().as_list()
        reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden3, layer4_weights) + layer4_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 1001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_num_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_num_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 50 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), test_num_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_num_labels))
