import functools
import os.path as op
import numpy as np
import random
import nibabel as nib
from sklearn.utils import shuffle
from glob import glob
import skimage.transform
import tensorflow as tf
import smtplib, ssl


def calltracker(func):
    """Initialize the decorator to track if a function has been called.

    :param func: The function to be wrapper
    :return: wrapper
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.called = True
        return func(*args, **kwargs)

    wrapper.called = False
    return wrapper


def data_summary(root_dir=None, dataset=None):
    """Provides the summary of the data. This is called in train_test_split function.

    :param root_dir: The directory of the processed data
    :param dataset: The specific data type ('healthy', 'ARDS' or 'all')
    :return: total_dataset (int), total_imgs (int)
    """
    if root_dir is None:
        root_dir = '/home/ubuntu/sl_root/Processed_data'
    else:
        root_dir = root_dir

    assert dataset == 'all' or dataset == 'healthy' or dataset == 'ARDS', 'Please select the correct dataset'

    # Set the processed data directory
    if dataset == 'all':
        processed_data_list = glob(op.join(root_dir, 'target_data_2/*'))
    elif dataset == 'ARDS':
        processed_data_list = glob(op.join(root_dir, 'target_data_2', '*ARDS*'))
    elif dataset == 'healthy':
        processed_data_list = glob(op.join(root_dir, 'target_data_2/*Healthy*'))

    total_dataset = len(processed_data_list)

    # Initialize empty list
    empty_list = []

    # Compute the total images
    for single_file in processed_data_list:
        img = nib.load(single_file).get_fdata()
        empty_list.append(img.shape[0])
    total_imgs = np.sum(empty_list)

    return total_dataset, total_imgs


def train_test_split(split_ratio=0.8, save_root_dir=None):
    """Performs the train test split.

    :param split_ratio: The split ratio of the training data to testing data
    :param save_root_dir: The directory of the processed data
    :return: train_ct, train_label_map, test_ct, test_label_map
    """
    # Set the processed data directories
    if save_root_dir is None:
        save_root_dir = '/home/ubuntu/sl_root/Processed_data'
    ct_dir = op.join(save_root_dir, 'target_data_2')
    label_map_dir = op.join(save_root_dir, 'source_data_2')

    # Obtain the healthy CT data and label map list
    healthy_ct_list = glob(op.join(ct_dir, '*Healthy*'))
    healthy_label_map_list = glob(op.join(label_map_dir, '*Healthy*'))

    # Obtain the ARDS CT data and label map list
    ARDS_ct_list = glob(op.join(ct_dir, '*ARDS*'))
    ARDS_label_map_list = glob(op.join(label_map_dir, '*ARDS*'))

    # Sort the list
    healthy_ct_list.sort(), healthy_label_map_list.sort()
    ARDS_ct_list.sort(), ARDS_label_map_list.sort()

    # Shuffle the CT data and label map list
    healthy_ct_list, healthy_label_map_list = shuffle(healthy_ct_list, healthy_label_map_list)
    ARDS_ct_list, ARDS_label_map_list = shuffle(ARDS_ct_list, ARDS_label_map_list)

    # Obtain the summary of the processed data
    print('\nObtaining the summary of the healthy dataset')
    healthy_total_dataset, healthy_total_imgs = data_summary(save_root_dir, dataset='healthy')
    healthy_average_imgs = round(healthy_total_imgs / healthy_total_dataset)
    print('Average of healthy dataset: ' + str(healthy_average_imgs) + '\n')

    print('Obtaining the summary of the ARDS dataset')
    ARDS_total_dataset, ARDS_total_imgs = data_summary(save_root_dir, dataset='ARDS')
    ARDS_average_imgs = round(ARDS_total_imgs / ARDS_total_dataset)
    print('Average of ARDS dataset: ' + str(ARDS_average_imgs) + '\n')

    # Create random index based on sampling
    print('Sampling ... ')
    counter = 0
    idx = random.sample(range(ARDS_total_dataset), healthy_total_dataset + counter)

    ARDS_ct_sublist, ARDS_label_map_sublist = [], []
    for i in idx:
        ARDS_single_ct, ARDS_single_label_map = ARDS_ct_list[i], ARDS_label_map_list[i]
        ARDS_ct_sublist.append(ARDS_single_ct)
        ARDS_label_map_sublist.append(ARDS_single_label_map)

    # Check that the LTRC and ARDS does not deviate by too much
    num_list = []
    for single_file in ARDS_ct_sublist:
        img = nib.load(single_file).get_fdata()
        num = img.shape[0]
        num_list.append(num)
    ARDS_avg_imgs = round(sum(num_list) / (len(ARDS_ct_sublist) - counter))

    epsilon = round(0.10 * healthy_average_imgs)

    # Check that the LTRC and ARDS does not deviate by an epsilon range
    while ARDS_avg_imgs >= healthy_average_imgs + epsilon or ARDS_avg_imgs <= healthy_average_imgs - epsilon:

        # Check random index based on sampling
        counter += 1
        print('Resampling ... [Counter] - ', counter)
        idx = random.sample(range(ARDS_total_dataset), healthy_total_dataset + counter)

        ARDS_ct_sublist, ARDS_label_map_sublist = [], []
        for i in idx:
            ARDS_single_ct, ARDS_single_label_map = ARDS_ct_list[i], ARDS_label_map_list[i]
            ARDS_ct_sublist.append(ARDS_single_ct)
            ARDS_label_map_sublist.append(ARDS_single_label_map)

        # Check that the LTRC and ARDS does not deviate by too much
        num_list = []
        for single_file in ARDS_ct_sublist:
            img = nib.load(single_file).get_fdata()
            num = img.shape[0]
            num_list.append(num)
        ARDS_avg_imgs = round(sum(num_list) / (len(ARDS_ct_sublist) - counter))
        print('ARDS average images (2): ', ARDS_avg_imgs)

    # Initialize empty list
    final_ct, final_label_map = [], []

    final_ct.extend(ARDS_ct_sublist)
    final_label_map.extend(ARDS_label_map_sublist)
    final_ct.extend(healthy_ct_list)
    final_label_map.extend(healthy_label_map_list)

    final_ct.sort(), final_label_map.sort()
    final_ct, final_label_map = shuffle(final_ct, final_label_map)

    # Obtain the train ratio
    train_ratio = round(split_ratio * len(final_ct))

    train_ct, train_label_map = final_ct[:train_ratio], final_label_map[:train_ratio]
    test_ct, test_label_map = final_ct[train_ratio:], final_label_map[train_ratio:]

    # Place the file name only in the list
    train_ct = [file.split('/')[-1] for file in train_ct]
    train_label_map = [file.split('/')[-1] for file in train_label_map]

    test_ct = [file.split('/')[-1] for file in test_ct]
    test_label_map = [file.split('/')[-1] for file in test_label_map]
    return train_ct, train_label_map, test_ct, test_label_map


def data_augmentation(X1, X2, batch_size):
    """Performs the data augmentation.

    :param X1: Label map
    :param X2: CT data
    :param batch_size:
    :return: train_ct, train_label_map, test_ct, test_label_map
    """
    # Set the seed and rotation angle
    seed_1 = np.random.uniform()
    seed_2 = np.random.randint(100)
    seed_3 = np.random.randint(100)
    rot_angle = np.random.randint(360)

    # Perform data augmentation
    if batch_size == 1:

        label_map_slice = np.squeeze(X1, axis=0)
        ct_slice = np.squeeze(X2, axis=0)

        # Rotate the image
        if seed_1 >= 0.5:
            ct_slice, label_map_slice = skimage.transform.rotate(ct_slice, rot_angle), skimage.transform.rotate(label_map_slice, rot_angle)
            # Remove the background due to rotation
            ct_slice[ct_slice == 0] = ct_slice.min()
            label_map_slice[label_map_slice == 0] = label_map_slice.min()

            # Flip the image (left right)
            if (seed_2 % 2) == 0:
                ct_slice, label_map_slice = tf.image.flip_left_right(ct_slice), tf.image.flip_left_right(label_map_slice)

                # Flip the image (up down)
                if (seed_3 % 2) == 0:
                    ct_slice, label_map_slice = tf.image.flip_up_down(ct_slice), tf.image.flip_up_down(label_map_slice)

        # Rename to X1 and X2
        X1 = label_map_slice[np.newaxis, :, :, :]
        X2 = ct_slice[np.newaxis, :, :, :]

    elif batch_size != 1:

        # Initialize empty array
        ct_slice_array = np.array([]).reshape(0, X2.shape[1], X2.shape[2], X2.shape[3])
        label_map_slice_array = np.array([]).reshape(0, X1.shape[1], X1.shape[2], X1.shape[3])

        for label_map_slice, ct_slice in zip(X1, X2):

            # Rotate the image
            if seed_1 >= 0.5:
                ct_slice, label_map_slice = skimage.transform.rotate(ct_slice, rot_angle), skimage.transform.rotate(label_map_slice, rot_angle)
                # Remove the background due to rotation
                ct_slice[ct_slice == 0] = ct_slice.min()
                label_map_slice[label_map_slice == 0] = label_map_slice.min()

                # Flip the image (left right)
                if (seed_2 % 2) == 0:
                    ct_slice, label_map_slice = tf.image.flip_left_right(ct_slice), tf.image.flip_left_right(label_map_slice)

                    # Flip the image (up down)
                    if (seed_3 % 2) == 0:
                        ct_slice, label_map_slice = tf.image.flip_up_down(ct_slice), tf.image.flip_up_down(label_map_slice)

            # Concatenate them in first axis
            ct_slice_array = np.concatenate((ct_slice_array, ct_slice[np.newaxis, :, :, :]), axis=0)
            label_map_slice_array = np.concatenate((label_map_slice_array, label_map_slice[np.newaxis, :, :, :]), axis=0)

        # Rename to X1 and X2
        X1 = label_map_slice_array
        X2 = ct_slice_array
    return X1, X2


def send_email(info, info_type):
    """Send email to track the progress.

    :param info: The patient ID
    :param info_type: Epoch, 'pat_id', 'data_processing'
    :return: None
    """
    port = 465  # For SSL
    smtp_server = 'smtp.gmail.com'
    my_email = 'slloo.AWS@gmail.com'
    password = 'QwertY8888'

    if info_type == 'epoch':
        message = """\
        Subject: AUTOMATIC EMAIL FROM AWS
    
        Epoch : {}""".format(info)

    elif info_type == 'pat_id':
        message = """\
        Subject: AUTOMATIC EMAIL FROM AWS

        Dataset (Train): {}""".format(info)

    elif info_type == 'data_processing':
        message = """\
        Subject: AUTOMATIC EMAIL FROM AWS

        Dataset (Process): {}""".format(info)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(my_email, password)
        server.sendmail(my_email, my_email, message)
    pass


def check_existing_data(patient_list, save_root_dir):
    """Check if the single patient has been processed.

    :param patient_list: List of the patient
    :param save_root_dir: The processed data directory
    :return: unprocessed_list
    """
    # Set the processed data directories
    ct_list = glob(op.join(save_root_dir, 'target_data_2', '*'))
    label_map_list = glob(op.join(save_root_dir, 'source_data_2', '*'))

    processed_ct_list = [single_ct.split('/')[-1].split('_')[2] for single_ct in ct_list]
    processed_label_list = [single_label.split('/')[-1].split('_')[2] for single_label in label_map_list]

    # Check if the patient are already processed
    processed_list = []
    for single_pat in patient_list:
        if single_pat in processed_ct_list and single_pat in processed_label_list:
            # Remove from the list
            processed_list.append(single_pat)

    # Obtain the patient list that are not processed
    unprocessed_list = [x for x in patient_list if x not in processed_list]
    return unprocessed_list
