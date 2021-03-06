from __future__ import print_function
import os
import sys
import tarfile
import pickle
from six.moves.urllib.request import urlretrieve
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import numpy as np
import digitStruct

DEBUG_MODE = False
FORCE_REBUILD = False

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


def maybe_download(url, expected_bytes, target_folder):
    filename = os.path.basename(url)
    full_target_path = os.path.join(target_folder, filename)
    print("attempting to download {}...".format(filename))
    if not os.path.exists(full_target_path):
        filename, _ = urlretrieve(url, full_target_path, reporthook=download_progress_hook)
    statinfo = os.stat(full_target_path)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return full_target_path


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    target_folder = os.path.dirname(filename)
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(path=target_folder)
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
        if DEBUG_MODE and len(data.keys()) >= 4:  # temporary, to keep fast
            break

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
    pixel_depth = 255.0
    num_labels = 7 + 10*5

    # remove images that don't exist
    real_data = [(filename, data) for filename, data in digit_data.items()
                 if os.path.exists(os.path.join(image_folder, filename))]

    num_samples = len(real_data)
    images = np.ndarray(dtype=np.float32, shape=(num_samples, 64, 64, 3))

    # labels are organized as follows:
    #   [0-6]: number of digits (hot-1).  0=0 digits, 1=1 digit, ... 6=more than 5
    #   [7-16]: digit 1 (hot-1)
    #   [17-26]: digit 2 (hot-1)
    #   ...
    labels = np.ndarray(dtype=np.float32, shape=(num_samples, num_labels))  # labels, hot 1 encoding
    image_index = 0
    for filename, data in real_data:
        pathname = os.path.join(image_folder, filename)
        print("loading image: {}".format(pathname))
        image_data = ndimage.imread(pathname).astype(float)

        if DEBUG_MODE:
            plt.imshow(image_data, interpolation=None)
            plt.show()

        image_data = 2.0 * image_data / pixel_depth - 1.0   # range goes from [0,255] -> [-1.0, 1.0]

        # find uber bounding box
        uber_bbox = get_uber_bounding_box(data)

        # crop
        cropped_data = image_data[
                       uber_bbox["top"]:uber_bbox["bottom"],
                       uber_bbox["left"]:uber_bbox["right"],
                       :]

        if DEBUG_MODE:
            # show the cropped image but put it back to 0-255 range
            debug_image = (cropped_data + 0.5) * pixel_depth
            plt.imshow(debug_image, interpolation=None)
            plt.show()

        # resize to 64x64
        resized_data = ndimage.zoom(cropped_data, (64.0/cropped_data.shape[0], 64.0/cropped_data.shape[1], 1.0),
                                    order=1, prefilter=False)
        # resized_data = cropped_data  # works
        # resized_data = ndimage.zoom(cropped_data, (0.995, 1.0, 1.0))

        if DEBUG_MODE:
            # show the cropped image but put it back to 0-1 range
            debug_image = resized_data + 0.5
            plt.imshow(debug_image)
            plt.show()

        # randomly crop to 54x54
        # r = np.random.randint(0, 9, 2)
        # crop2_data = resized_data[r[0]:r[0]+54, r[1]:r[1]+54]
        # images[image_index, :, :, :] = crop2_data

        images[image_index, :, :, :] = resized_data

        label = np.zeros(shape=num_labels)
        label[len(data) if len(data) < 6 else 6] = 1.0  # num digits
        if 0 < len(data) < 6:
            for d in range(len(data)):
                digit = data[d]['digit']
                if digit in range(10):  # sometimes the data might have "10"... just count it as "0"
                    label[7 + d*10 + data[d]['digit']] = 1.0  # digit d

        labels[image_index, :] = label

        if DEBUG_MODE:
            print("label details:")
            print("  num digits: {}".format(label[0:7]))
            for i in range(5):
                print("  digit {d}: ({digit}), {hot_one}".format(
                    d=i + 1,
                    digit="?",  #[np.where(r == 1)[0] for r in label[(7 + 10*i):(17 + 10*i)]],
                    hot_one=label[(7 + 10*i):(17 + 10*i)]
                ))

        image_index += 1

    return images, labels


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

pickle_filename = 'data/dataset.pickle'

if not os.path.exists('data'):
    os.mkdir('data')
train_archive = maybe_download('http://ufldl.stanford.edu/housenumbers/train.tar.gz', 404141560, 'data')
test_archive = maybe_download('http://ufldl.stanford.edu/housenumbers/test.tar.gz', 276555967, 'data')


# extract and split the test data into test and validation
train_folder = maybe_extract(train_archive)
test_folder = maybe_extract(test_archive)
valid_folder = "data/validation"
if not os.path.exists(valid_folder):
    os.makedirs(valid_folder)
    png_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if os.path.splitext(f)[1] == ".png"]
    for f in png_files[:10000]:
        os.rename(f, f.replace(test_folder, valid_folder))

dataset = get_data_from_pickle()
if FORCE_REBUILD or not dataset:
    print("No pickle data...")
    train_data = load_digit_data('data/train/digitStruct.mat')
    test_data = load_digit_data('data/test/digitStruct.mat')

    train_images, train_labels = load_images(train_folder, train_data)
    valid_images, valid_labels = load_images(valid_folder, test_data)  # the data is in test_data for both
    test_images, test_labels = load_images(test_folder, test_data)


    dataset = {
        'train_images': train_images,
        'train_labels': train_labels,
        'valid_images': valid_images,
        'valid_labels': valid_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }
    print("Picking data...")
    pickle_data(dataset)
else:
    print("Found pickle data...")
    train_images = dataset['train_images']
    train_labels = dataset['train_labels']
    valid_images = dataset['valid_images']
    valid_labels = dataset['valid_labels']
    test_images = dataset['test_images']
    test_labels = dataset['test_labels']
    print("Training images: {}".format(len(train_images)))
    print("Training labels: {}".format(len(train_labels)))
    print("Validation images: {}".format(len(valid_images)))
    print("Validation labels: {}".format(len(valid_labels)))
    print("Test images: {}".format(len(test_images)))
    print("Test labels: {}".format(len(test_labels)))


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


image_size = 64
num_labels = 7 + 10*5
batch_size = 16
patch_size = 5
num_hidden = 64
num_channels = 3

graph = tf.Graph()
with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_images)
    tf_test_dataset = tf.constant(test_images)
    # single dataset for inference
    tf_single_dataset = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels), name="dataset")

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, 16], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([16]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 16, 16], stddev=0.1))
    layer2_biases = tf.Variable(tf.zeros([16]))
    layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 16, 32], stddev=0.1))
    layer3_biases = tf.Variable(tf.zeros([32]))
    layer4_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 32, 64], stddev=0.1))
    layer4_biases = tf.Variable(tf.zeros([64]))
    layer5_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, 64, 64], stddev=0.1))
    layer5_biases = tf.Variable(tf.zeros([64]))

    layer6_weights = tf.Variable(tf.truncated_normal([image_size // 8 * image_size // 8 * 64, num_hidden], stddev=0.1))
    layer6_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer7_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer7_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        # Convolution
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(conv, layer2_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(tf.nn.relu(conv + layer2_biases), [1, 4, 4, 1], [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(conv, layer3_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.max_pool(tf.nn.relu(conv + layer3_biases), [1, 4, 4, 1], [1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(conv, layer4_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.relu(conv + layer4_biases)
        conv = tf.nn.conv2d(conv, layer5_weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.relu(conv + layer5_biases)
        conv = tf.nn.dropout(conv, 0.5)

        # Fully Connected Layer
        shape = conv.get_shape().as_list()
        fc = tf.reshape(conv, [shape[0], shape[1] * shape[2] * shape[3]])
        fc = tf.nn.relu(tf.matmul(fc, layer6_weights) + layer6_biases)

        return tf.nn.xw_plus_b(fc, layer7_weights, layer7_biases)


    # Training computation.
    logits = model(tf_train_dataset)
    
    # make a vector that we can multiply the losses by (we should ignore any digit that "doesn't exist")
    num_digits_vector = [
        tf.reduce_sum(tf_train_labels[:, 1:7]),
        tf.reduce_sum(tf_train_labels[:, 2:7]),
        tf.reduce_sum(tf_train_labels[:, 3:7]),
        tf.reduce_sum(tf_train_labels[:, 4:7]),
        tf.reduce_sum(tf_train_labels[:, 5:7])
    ]

    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[:, 0:7], logits=logits[:, 0:7]) +
        tf.multiply(num_digits_vector[0], tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[:, 7:17], logits=logits[:, 7:17])) +
        tf.multiply(num_digits_vector[1], tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[:, 17:27], logits=logits[:, 17:27])) +
        tf.multiply(num_digits_vector[2], tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[:, 27:37], logits=logits[:, 27:37])) +
        tf.multiply(num_digits_vector[3], tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[:, 37:47], logits=logits[:, 37:47])) +
        tf.multiply(num_digits_vector[4], tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels[:, 47:57], logits=logits[:, 47:57]))
    )

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
    single_prediction = tf.nn.softmax(model(tf_single_dataset), name="model")

num_steps = 50000

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 50 == 0:
            print('Minibatch step {s}: loss: {l}, accuracy: {a}, validation accuracy:{v}'.format(
                s=step,
                l=l,
                a=accuracy(predictions, batch_labels),
                v=accuracy(valid_prediction.eval(), valid_labels)))

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    saver = tf.train.Saver()
    inference_folder = os.path.join(os.path.dirname(__file__), "inference",
                                    time.strftime("%Y-%m-%d-%H-%M", time.gmtime()))
    if not os.path.exists(inference_folder):
        os.mkdir(inference_folder)
    saver.save(session, os.path.join(inference_folder, "model"))