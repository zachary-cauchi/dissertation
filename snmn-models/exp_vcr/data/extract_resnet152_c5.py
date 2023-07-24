import argparse
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
parser.add_argument('--res', type=int, default=8,
    help='Feature size resxres. 8 Will produce 8x8 features, etc')
args = parser.parse_args()

gpu_id = args.gpu_id  # set GPU id to use
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

resnet152_model = '../tfmodel/resnet/resnet_v1_152.tfmodel'
image_basedir = '../vcr_dataset/vcr1images/'
save_basedir = f'./resnet152_c5_{args.res}x{args.res}/'
H = 64 * args.res
W = 64 * args.res

image_batch = tf.placeholder(tf.float32, [1, H, W, 3])
resnet152_c5 = resnet_v1.resnet_v1_152_c5(image_batch, is_training=False)
resnet152_c5_8x8 = tf.nn.avg_pool(
    resnet152_c5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
saver = tf.train.Saver()
sess = tf.Session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver.restore(sess, resnet152_model)


def extract_image_resnet152_c5(impath):
    im = skimage.io.imread(impath)
    if im.ndim == 2:  # Gray 2 RGB
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    im = im[..., :3]
    assert im.dtype == np.uint8
    im = skimage.transform.resize(im, [H, W], preserve_range=True)
    im_val = (im[np.newaxis, ...] - channel_mean)
    resnet152_c5_8x8_val = resnet152_c5_8x8.eval({image_batch: im_val}, sess)
    return resnet152_c5_8x8_val


def extract_dataset_resnet152_c5(image_dir, save_dir, ext_filter='*.png'):
    image_list = glob(image_dir + '/' + ext_filter)
    os.makedirs(save_dir, exist_ok=True)

    for n_im, impath in enumerate(image_list):
        if (n_im+1) % 100 == 0:
            print('processing %d / %d' % (n_im+1, len(image_list)))
        image_name = os.path.basename(impath).rsplit('.', 1)[0]
        save_path = os.path.join(save_dir, image_name + '.npy')
        if not os.path.exists(save_path):
            resnet152_c5_val = extract_image_resnet152_c5(impath)
            np.save(save_path, resnet152_c5_val)

image_sets = [dir for dir in os.listdir(image_basedir) if os.path.isdir(os.path.join(image_basedir, dir))]
set_count = len(image_sets)
set_count_digits = len("%i" % set_count)

for i, image_set in enumerate(image_sets):
    print(f'({i + 1:0{set_count_digits}}/{set_count}) Extracting image set {image_set}')
    extract_dataset_resnet152_c5(
        os.path.join(image_basedir, image_set),
        os.path.join(save_basedir, image_set),
        ext_filter='*.jpg')
    print('Done.')
