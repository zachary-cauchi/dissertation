import argparse
import tqdm
import os
import sys; sys.path.append('../../')  # NOQA
from glob import glob
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf

from util.nets import resnet_v1, channel_mean


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--file_type', type=str, choices=['tfrecord', 'npy'], default='npy')
parser.add_argument('--res', type=int, default=8,
    help='Feature size resxres. 8 Will produce 8x8 features, etc')
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
file_type = args.file_type

resnet152_model = '../tfmodel/resnet/resnet_v1_152.tfmodel'
image_basedir = '../vcr_dataset/vcr1images/'
save_basedir = f'./resnet152_c5_{args.res}x{args.res}/' if file_type == 'npy' else f'./tfrecords_resnet152_c5_{args.res}x{args.res}/'
H = 64 * args.res
W = 64 * args.res

image_batch = tf.placeholder(tf.float32, [1, H, W, 3])
resnet152_c5 = resnet_v1.resnet_v1_152_c5(image_batch, is_training=False)
resnet152_c5_8x8 = tf.nn.avg_pool(
    resnet152_c5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
resnet152_c5_8x8_serialized = tf.io.serialize_tensor(resnet152_c5_8x8)
saver = tf.train.Saver()
sess = tf.Session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver.restore(sess, resnet152_model)


def extract_image_resnet152_c5(impath, serialize=False):
    im = skimage.io.imread(impath)
    if im.ndim == 2:  # Gray 2 RGB
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    im = im[..., :3]
    assert im.dtype == np.uint8
    im = skimage.transform.resize(im, [H, W], preserve_range=True)
    im_val = (im[np.newaxis, ...] - channel_mean)
    if serialize:
        resnet152_c5_8x8_val = resnet152_c5_8x8_serialized.eval({ image_batch: im_val }, sess)
    else:
        resnet152_c5_8x8_val = resnet152_c5_8x8.eval({image_batch: im_val}, sess)
    return resnet152_c5_8x8_val


def extract_dataset_resnet152_c5_npy(image_dir, save_dir, ext_filter='*.png'):
    image_list = glob(image_dir + '/' + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for impath in tqdm.tqdm(image_list, leave=False):
        image_name = os.path.basename(impath).rsplit('.', 1)[0]
        save_path = os.path.join(save_dir, image_name + '.npy')
        if not os.path.exists(save_path):
            resnet152_c5_val = extract_image_resnet152_c5(impath)
            np.save(save_path, resnet152_c5_val)

def extract_dataset_resnet152_c5_tfrecord(image_dir, save_dir, set_name, ext_filter='*.png'):
    image_list = glob(image_dir + '/' + ext_filter)
    save_path = os.path.join(save_dir, set_name)
    os.makedirs(save_path, exist_ok=True)

    features = {}
    for impath in tqdm.tqdm(image_list, leave=False):
        image_name = os.path.basename(impath).rsplit('.', 1)[0]
        features[image_name] = extract_image_resnet152_c5(impath, serialize=True)

    create_tfrecord_file(save_path, features)

def create_tfrecord_file(save_dir, image_set: dict):
    for key, array in tqdm.tqdm(image_set.items(), leave=False):
        with tf.python_io.TFRecordWriter(os.path.join(save_dir, key + '.tfrecords')) as writer:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={ 'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[array])) }
                )
            )

            writer.write(example.SerializeToString())

image_sets = [dir for dir in os.listdir(image_basedir) if os.path.isdir(os.path.join(image_basedir, dir))]
set_count = len(image_sets)
set_count_digits = len("%i" % set_count)

for image_set in tqdm.tqdm(image_sets, unit='Image sets'):
    if file_type == 'npy':
        extract_dataset_resnet152_c5_npy(
            os.path.join(image_basedir, image_set),
            os.path.join(save_basedir, image_set),
            ext_filter='*.jpg')
    else:
        extract_dataset_resnet152_c5_tfrecord(
            os.path.join(image_basedir, image_set),
            os.path.join(save_basedir),
            image_set,
            ext_filter='*.jpg')
print('All features extracted and saved.')
