import numpy as np
import cv2
import random
from scipy.ndimage import rotate

size = 100
clahe_parameter = int(255 / 10)  # 10 histograms


def np_transform_bgr(a):  # images are in bgr format!!!
    # Transforms the data into standard rgb form (3, x, x)
    r = a[0, :, :]
    g = a[1, :, :]
    b = a[2, :, :]
    rgb = np.dstack((r, g, b))
    return rgb


def np_transform_rgb_inv(img):
    shape = np.shape(img)
    new_img = np.zeros(shape[::-1])
    new_img[0, :, :] = img[:, :, 0]
    new_img[1, :, :] = img[:, :, 1]
    new_img[2, :, :] = img[:, :, 2]
    return new_img.astype("uint8")


def rgb_img(a):  # images are in bgr format!!!
    # Transforms the data into standard rgb form (3, x, x)
    return a[0:3, :, :]


def rgb_grey(a):
    r = a[0, :, :]
    g = a[1, :, :]
    b = a[2, :, :]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def one_hot_image(a):
    # Seperates the one-hot-encoded part of the data
    return a[3, :, :]


# resizing the image
def image_resize(image, size):
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


# flips the imae
def random_flipper_horizontal(train_image):
    randomizer = [random.randint(0, 1) for x in
                  range(len(train_image))]  # creates boolean list and assigns true or false for each image
    for index in range(len(train_image)):
        image = train_image[index]
        if randomizer == True:  # if true then the element will be flipped
            flipped_image = np.fliplr(test_image)
            train_image[index] = flipped_image

    return train_image


def shear_crop_resize(image, size):  # size has to be tuple
    shear = random.uniform(-0.2, 0.2)  # introduce random effect to introduce flexibility to the input
    afine_tf = tf.AffineTransform(shear=shear)
    # Apply transform to image data
    modified = tf.warp(image, inverse_map=afine_tf)  # perform affine transformation
    col = image.shape[0]
    factor = int(shear * col)
    if shear >= 0:
        cropped_img = modified[int(factor / 2):col - int(factor / 2),
                      int(shear * col):col]  # crop image to remove part where it is black due to affine transformation
    else:
        row = col
        cropped_img = modified[0 - int(factor / 2):row + int(factor / 2), 0:col + (factor)]
    resized_img = image_resize(cropped_img, size)  # resize image to final size
    return resized_img


def clahe_4_rgb(
        input_img):  # clahe normalization for rgb images wwhere value of hsv colorroom was used to mimic gra channel value
    hsvImg = cv2.cvtColor(np.float32(input_img), cv2.COLOR_BGR2HSV)
    value_channel = hsvImg[:, :, 2]
    value_channel = np.uint16(value_channel * 255)
    clahe = cv2.createCLAHE(clipLimit=2)
    value_channel = np.uint16(value_channel * 255) + 30
    final_img = clahe.apply(value_channel)
    final_img = final_img / 255
    hsvImg[:, :, 2] = final_img
    rgb = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    rgb = np.int16(rgb)
    return rgb


def clahe_4_gray(input_img,
                 clipLimit):  # clahe normalization for gray images clip Limit could be hyperparameter if it shows significant changes to performance
    gray = cv2.cvtColor(np.float32(test_image), cv2.COLOR_BGR2GRAY)
    gray = np.uint16(gray * 255)
    clahe = cv2.createCLAHE(clipLimit=clipLimit)
    final_img = clahe.apply(gray)
    final_img = np.int16(final_img)
    del clahe
    return final_img


# def pre_process(data):


# def one_trans(img):
#    new_img = np.zeros((9, 256, 256))
#    for i in range(9):
#        new_img[i,:,:] = (i == img[0,:,:])
#    return new_img


def one_trans(img):
    new_img = np.zeros((9, np.shape(img)[1], np.shape(img)[2]))
    for i in range(9):
        new_img[i, :, :] = (i == img[:, :]).astype(float)
    return new_img


def one_trans_inv(img):
    new_img = np.zeros(np.shape(img))
    for i in range(9):
        new_img[:, :] = img[0, i, :, :] * i
    return new_img


def random_crop(img, crop_size=(150, 150), seed=1):
    np.random.seed(seed)
    img_shape = img.shape
    pad_size = (img_shape[0] - crop_size[0]) // 2
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h - crop_size[0]), np.random.randint(w - crop_size[1])
    img = img[y:y + crop_size[0], x:x + crop_size[1]]

    return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')


def rotate_img(img, angle, bg_patch=(0, 0)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0, 1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


# rotating the image a random amount of times
def ranodom_rotation(batch):
    # rotating the image randomly (only 0 deg, 90 deg, 180 deg and 270 deg for keeping the same informations)
    rand_rot = (random.randint(0, 3))
    for i in range(0, rand_rot):
        if rand_rot == 0:
            return batch
        else:
            batch = np.rot90(batch, 1)
    return batch