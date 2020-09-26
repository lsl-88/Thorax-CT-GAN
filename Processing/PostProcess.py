from utils import *

import os
import os.path as op
import numpy as np
from skimage.transform import resize
import nibabel as nib


class PostProcess:
    """PostProcessing of the CT data and label map from the binary maps."""
    def __init__(self, save_root_dir=None, merge_channel=None):

        if save_root_dir is None:
            self.save_root_dir = '/home/ubuntu/sl_root/Processed_data'
        else:
            self.save_root_dir = save_root_dir
        self.merge_channel = merge_channel

        # Variables to store data
        self.save_dir_target = None
        self.save_dir_source = None

        self.create_save_directories()

    def __repr__(self):
        return '{self.__class__.__name__}(save_root_dir={self.save_root_dir})'

    def create_save_directories(self):
        """Creates the save directories for the postprocessed data.

        :return: None
        """
        # Set the directories name
        self.save_dir_target = op.join(self.save_root_dir, 'target_data_2')
        self.save_dir_source = op.join(self.save_root_dir, 'source_data_2')

        # Create the ct_data and label maps directories
        if not op.exists(self.save_root_dir):
            os.mkdir(self.save_root_dir)
        if not op.exists(self.save_dir_target):
            os.mkdir(self.save_dir_target)
        if not op.exists(self.save_dir_source):
            os.mkdir(self.save_dir_source)
        return self

    @calltracker
    def LTRC_normalization(self, pat_obj):
        """Performs the normalization to [-1,1] for the LTRC source data and target data.

        :param pat_obj: Object of LTRC class
        :return: source_data_norm (LTRC), target_data_norm (LTRC)
        """
        # Initialize empty dictionary
        source_data_norm, target_data_norm = {}, {}

        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.LTRC_resize.called:
            print('Performing normalization on resized LTRC data')
        else:
            print('Performing normalization on original size LTRC data')

        for series_num, series_data in source_data.items():

            # Initialize empty array
            src_array = np.array([]).reshape(series_data.shape[0], series_data.shape[1], series_data.shape[2], 0)

            for channel in range(series_data.shape[3]):
                src = series_data[:, :, :, channel]

                # Normalize source data to [-1, 1]
                src = np.float32(src)
                src = (src - (src.max() / 2)) / (src.max() / 2)

                src_array = np.concatenate((src_array, src[:, :, :, np.newaxis]), axis=3)
            source_data_norm[series_num] = src_array

        for series_num, series_data in target_data.items():

            # Initialize empty array
            tgt_array = np.array([]).reshape(series_data.shape[0], series_data.shape[1], series_data.shape[2], 0)

            for channel in range(series_data.shape[3]):
                tgt = series_data[:, :, :, channel]

                # Normalize target data to [-1, 1]
                tgt = np.float32(tgt)
                tgt[tgt < -1024] = -1024
                tgt[tgt < 0] = tgt[tgt < 0] / (-tgt.min())
                tgt[tgt > 0] = tgt[tgt > 0] / (tgt.max())

                tgt_array = np.concatenate((tgt_array, tgt[:, :, :, np.newaxis]), axis=3)
            target_data_norm[series_num] = tgt_array

        # Cache the data
        pat_obj.source_data = source_data_norm
        pat_obj.target_data = target_data_norm
        return source_data_norm, target_data_norm

    @calltracker
    def UMM_normalization(self, pat_obj):
        """Performs the normalization to [-1,1] for the UMM source data and target data.

        :param pat_obj: Object of UMM class
        :return: source_data_norm (UMM), target_data_norm (UMM)
        """
        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.UMM_resize.called:
            print('Performing normalization on resized UMM data')
        else:
            print('Performing normalization on original size UMM data')

        if not isinstance(source_data, dict):

            # Initialize empty array
            src_array = np.array([]).reshape(source_data.shape[0], source_data.shape[1], source_data.shape[2], 0)
            tgt_array = np.array([]).reshape(target_data.shape[0], target_data.shape[1], target_data.shape[2], 0)

            for channel in range(source_data.shape[3]):
                src = source_data[:, :, :, channel]

                # Normalize source data to [-1, 1]
                src = np.float32(src)
                src = (src - (src.max() / 2)) / (src.max() / 2)
                src_array = np.concatenate((src_array, src[:, :, :, np.newaxis]), axis=3)
            source_data_norm = src_array

            for channel in range(target_data.shape[3]):
                tgt = target_data[:, :, :, channel]

                # Normalize target data to [-1, 1]
                tgt = np.float32(tgt)
                tgt[tgt < -1024] = -1024
                tgt[tgt < 0] = tgt[tgt < 0] / (-tgt.min())
                tgt[tgt > 0] = tgt[tgt > 0] / (tgt.max())
                tgt_array = np.concatenate((tgt_array, tgt[:, :, :, np.newaxis]), axis=3)
            target_data_norm = tgt_array

            # Cache the data
            pat_obj.source_data = source_data_norm
            pat_obj.target_data = target_data_norm
        else:
            source_data_norm, target_data_norm = {}, {}

            for series_date, series_data in source_data.items():

                # Initialize empty array
                src_array = np.array([]).reshape(series_data.shape[0], series_data.shape[1], series_data.shape[2], 0)

                for channel in range(series_data.shape[3]):
                    src = series_data[:, :, :, channel]

                    # Normalize source data to [-1, 1]
                    src = np.float32(src)
                    src = (src - (src.max() / 2)) / (src.max() / 2)
                    src_array = np.concatenate((src_array, src[:, :, :, np.newaxis]), axis=3)
                source_data_norm[series_date] = src_array

            for series_date, series_data in target_data.items():

                # Initialize empty array
                tgt_array = np.array([]).reshape(series_data.shape[0], series_data.shape[1], series_data.shape[2], 0)

                for channel in range(series_data.shape[3]):
                    tgt = series_data[:, :, :, channel]

                    # Normalize target data to [-1, 1]
                    tgt = np.float32(tgt)
                    tgt[tgt < -1024] = -1024
                    tgt[tgt < 0] = tgt[tgt < 0] / (-tgt.min())
                    tgt[tgt > 0] = tgt[tgt > 0] / (tgt.max())

                    tgt_array = np.concatenate((tgt_array, tgt[:, :, :, np.newaxis]), axis=3)
                target_data_norm[series_date] = tgt_array

            # Cache the data
            pat_obj.source_data = source_data_norm
            pat_obj.target_data = target_data_norm
        return source_data_norm, target_data_norm

    @calltracker
    def UKSH_normalization(self, pat_obj):
        """Performs the normalization to [-1,1] for the UKSH source data and target data.

        :param pat_obj: Object of UKSH class
        :return: source_data_norm (UKSH), target_data_norm (UKSH)
        """
        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.UKSH_resize.called:
            print('Performing normalization on resized UKSH data')
        else:
            print('Performing normalization on original size UKSH data')

        # Initialize empty array
        src_array = np.array([]).reshape(source_data.shape[0], source_data.shape[1], source_data.shape[2], 0)
        tgt_array = np.array([]).reshape(target_data.shape[0], target_data.shape[1], target_data.shape[2], 0)

        for channel in range(source_data.shape[3]):
            src = source_data[:, :, :, channel]

            # Normalize source data to [-1, 1]
            src = np.float32(src)
            src = (src - (src.max() / 2)) / (src.max() / 2)
            src_array = np.concatenate((src_array, src[:, :, :, np.newaxis]), axis=3)
        source_data_norm = src_array

        for channel in range(target_data.shape[3]):
            # Normalize target data to [-1, 1]
            tgt = target_data[:, :, :, channel]
            tgt = np.float32(tgt)
            tgt[tgt < -1024] = -1024
            tgt[tgt < 0] = tgt[tgt < 0] / (-tgt.min())
            tgt[tgt > 0] = tgt[tgt > 0] / (tgt.max())
            tgt_array = np.concatenate((tgt_array, tgt[:, :, :, np.newaxis]), axis=3)
        target_data_norm = tgt_array

        # Cache the data
        pat_obj.source_data = source_data_norm
        pat_obj.target_data = target_data_norm
        return source_data_norm, target_data_norm

    @calltracker
    def LTRC_resize(self, pat_obj, size=None):
        """Resize the LTRC source data and target data.

        :param pat_obj: Object of LTRC class
        :param size: Size of the image to resize to
        :return: source_data_resized (LTRC), target_data_resized (LTRC)
        """
        # Initialize empty dictionary
        source_data_resized, target_data_resized = {}, {}

        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.LTRC_normalization is False:
            print('Performing resizing on unnormalized LTRC data')
        else:
            print('Performing resizing on normalized LTRC data')

        for series_num, series_data in source_data.items():

            # Initialize empty array
            src_array = np.array([]).reshape(0, series_data.shape[1], series_data.shape[2], series_data.shape[0])

            for channel in range(series_data.shape[3]):
                src = series_data[:, :, :, channel]
                src = np.rollaxis(src, axis=0, start=2)

                # Resize the image
                src = resize(src, output_shape=size, anti_aliasing=True)

                src_array = np.concatenate((src_array, src), axis=0)

                # Roll the axis to (num samples, ht, wt, ch)
                src_array = np.rollaxis(src_array, axis=0, start=3)
                src_array = np.rollaxis(src_array, axis=3, start=0)

            source_data_resized[series_num] = src_array

        for series_num, series_data in target_data.items():

            # Initialize empty array
            tgt_array = np.array([]).reshape(0, series_data.shape[1], series_data.shape[2], series_data.shape[0])

            for channel in range(series_data.shape[3]):
                tgt = series_data[:, :, :, channel]
                tgt = np.rollaxis(tgt, axis=0, start=2)

                # Resize the label map
                tgt = resize(tgt, output_shape=size, anti_aliasing=True)

                tgt_array = np.concatenate((tgt_array, tgt), axis=0)

                # Roll the axis to (num samples, ht, wt, ch)
                tgt_array = np.rollaxis(tgt_array, axis=0, start=3)
                tgt_array = np.rollaxis(tgt_array, axis=3, start=0)

            target_data_resized[series_num] = tgt_array

        # Cache the data
        pat_obj.source_data = source_data_resized
        pat_obj.target_data = target_data_resized
        return source_data_resized, target_data_resized

    @calltracker
    def UMM_resize(self, pat_obj, size=None):
        """Resize the UMM source data and target data.

        :param pat_obj: Object of UMM class
        :param size: Size of the image to resize to
        :return: source_data_resized (UMM), target_data_resized (UMM)
        """
        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.UMM_normalization is False:
            print('Performing resizing on unnormalized UMM data')
        else:
            print('Performing resizing on normalized UMM data')

        if not isinstance(source_data, dict):

            # Initialize empty array
            src_array = np.array([]).reshape(0, source_data.shape[1], source_data.shape[2], source_data.shape[0])
            tgt_array = np.array([]).reshape(0, target_data.shape[1], target_data.shape[2], target_data.shape[0])

            for channel in range(source_data.shape[3]):
                src = source_data[:, :, :, channel]
                src = np.rollaxis(src, axis=0, start=2)

                # Resize the image
                src = resize(src, output_shape=size, anti_aliasing=True)

                src_array = np.concatenate((src_array, src), axis=0)

                # Roll the axis to (num samples, ht, wt, ch)
                src_array = np.rollaxis(src_array, axis=0, start=3)
                src_array = np.rollaxis(src_array, axis=3, start=0)

            source_data_resized = src_array

            for channel in range(target_data.shape[3]):
                tgt = target_data[:, :, :, channel]
                tgt = np.rollaxis(tgt, axis=0, start=2)

                # Resize the label map
                tgt = resize(tgt, output_shape=size, anti_aliasing=True)

                tgt_array = np.concatenate((tgt_array, tgt), axis=0)

                # Roll the axis to (num samples, ht, wt, ch)
                tgt_array = np.rollaxis(tgt_array, axis=0, start=3)
                tgt_array = np.rollaxis(tgt_array, axis=3, start=0)

            target_data_resized = tgt_array

            # Cache the data
            pat_obj.source_data = source_data_resized
            pat_obj.target_data = target_data_resized
        else:
            source_data_resized, target_data_resized = {}, {}

            for series_date, series_data in source_data.items():

                # Initialize empty array
                src_array = np.array([]).reshape(0, series_data.shape[1], series_data.shape[2], series_data.shape[0])

                for channel in range(series_data.shape[3]):
                    src = series_data[:, :, :, channel]
                    src = np.rollaxis(src, axis=0, start=2)

                    # Resize the image
                    src = resize(src, output_shape=size, anti_aliasing=True)

                    src_array = np.concatenate((src_array, src), axis=0)

                    # Roll the axis to (num samples, ht, wt, ch)
                    src_array = np.rollaxis(src_array, axis=0, start=3)
                    src_array = np.rollaxis(src_array, axis=3, start=0)

                source_data_resized[series_date] = src_array

            for series_date, series_data in target_data.items():

                # Initialize empty array
                tgt_array = np.array([]).reshape(0, series_data.shape[1], series_data.shape[2], series_data.shape[0])

                for channel in range(series_data.shape[3]):
                    tgt = series_data[:, :, :, channel]
                    tgt = np.rollaxis(tgt, axis=0, start=2)

                    # Resize the label map
                    tgt = resize(tgt, output_shape=size, anti_aliasing=True)

                    tgt_array = np.concatenate((tgt_array, tgt), axis=0)

                    # Roll the axis to (num samples, ht, wt, ch)
                    tgt_array = np.rollaxis(tgt_array, axis=0, start=3)
                    tgt_array = np.rollaxis(tgt_array, axis=3, start=0)

                target_data_resized[series_date] = tgt_array

            # Cache the data
            pat_obj.source_data = source_data_resized
            pat_obj.target_data = target_data_resized
        return source_data_resized, target_data_resized

    @calltracker
    def UKSH_resize(self, pat_obj, size=None):
        """Resize the UKSH source data and target data.

        :param pat_obj: Object of UKSH class
        :param size: Size of the image to resize to
        :return: source_data_resized (UKSH), target_data_resized (UKSH)
        """
        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.UKSH_normalization is False:
            print('Performing resizing on unnormalized UKSH data')
        else:
            print('Performing resizing on normalized UKSH data')

        # Initialize empty array
        src_array = np.array([]).reshape(0, source_data.shape[1], source_data.shape[2], source_data.shape[0])
        tgt_array = np.array([]).reshape(0, target_data.shape[1], target_data.shape[2], target_data.shape[0])

        for channel in range(source_data.shape[3]):
            src = source_data[:, :, :, channel]
            src = np.rollaxis(src, axis=0, start=2)

            # Resize the image
            src = resize(src, output_shape=size, anti_aliasing=True)

            src_array = np.concatenate((src_array, src), axis=0)

            # Roll the axis to (num samples, ht, wt, ch)
            src_array = np.rollaxis(src_array, axis=0, start=3)
            src_array = np.rollaxis(src_array, axis=3, start=0)

        source_data_resized = src_array

        for channel in range(target_data.shape[3]):
            tgt = target_data[:, :, :, channel]
            tgt = np.rollaxis(tgt, axis=0, start=2)

            # Resize the label map
            tgt = resize(tgt, output_shape=size, anti_aliasing=True)

            tgt_array = np.concatenate((tgt_array, tgt), axis=0)

            # Roll the axis to (num samples, ht, wt, ch)
            tgt_array = np.rollaxis(tgt_array, axis=0, start=3)
            tgt_array = np.rollaxis(tgt_array, axis=3, start=0)

        target_data_resized = tgt_array

        # Cache the data
        pat_obj.source_data = source_data_resized
        pat_obj.target_data = target_data_resized
        return source_data_resized, target_data_resized

    def LTRC_remove_slices(self, pat_obj, percentage):
        """Removes the slices of LTRC source data and target data that are less than the percentage of the
         background.

        :param pat_obj: Object of LTRC class
        :param percentage: Percentage threshold
        :return: source_data_dict (LTRC), target_data_dict (LTRC)
        """
        # Initialize empty dictionaries
        index_dict, source_data_dict, target_data_dict = {}, {}, {}

        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        print('Removing Lungless data in the second channel.')

        for series_num, series_data in source_data.items():

            # Read the 2nd channel (Lungs only)
            src = series_data[:, :, :, 1]

            # Initialize empty list
            series_index = []

            # Total number of pixels
            total_pixels = src.shape[1] * src.shape[2]

            for index, single_slice in enumerate(src):

                # Number of pixels that are lungs
                num_pixels = single_slice[single_slice != single_slice.min()]

                if len(num_pixels) >= (percentage * total_pixels):
                    series_index.append(index)
            index_dict[series_num] = series_index

        for series_num, series_data in source_data.items():

            series_index = index_dict[series_num]
            tgt_series_data = target_data[series_num]

            # Initialize empty array
            src_array = np.array([]).reshape(len(series_index), series_data.shape[1], series_data.shape[2], 0)
            tgt_array = np.array([]).reshape(len(series_index), tgt_series_data.shape[1], tgt_series_data.shape[2], 0)

            for channel in range(series_data.shape[3]):
                src_series_ch_data = series_data[:, :, :, channel]
                tgt_series_ch_data = tgt_series_data[:, :, :, channel]
                src_ch_data = src_series_ch_data[series_index]
                tgt_ch_data = tgt_series_ch_data[series_index]

                src_array = np.concatenate((src_array, src_ch_data[:, :, :, np.newaxis]), axis=3)
                tgt_array = np.concatenate((tgt_array, tgt_ch_data[:, :, :, np.newaxis]), axis=3)

            source_data_dict[series_num] = src_array
            target_data_dict[series_num] = tgt_array

        # Cache the data
        pat_obj.source_data = source_data_dict
        pat_obj.target_data = target_data_dict
        return source_data_dict, target_data_dict

    def UMM_remove_slices(self, pat_obj, percentage):
        """Removes the slices of UMM source data and target data that are less than the percentage of the
         background.

        :param pat_obj: Object of UMM class
        :param percentage: Percentage threshold
        :return: source_data_dict (UMM), target_data_dict (UMM)
        """
        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        print('Removing Lungless data in the second channel.')

        if not isinstance(source_data, dict):

            src = source_data[:, :, :, 1]

            # Initialize empty list
            series_index = []

            # Total number of pixels
            total_pixels = src.shape[1] * src.shape[2]

            for index, single_slice in enumerate(src):

                # Number of pixels that are lungs
                num_pixels = single_slice[single_slice != single_slice.min()]

                if len(num_pixels) >= (percentage * total_pixels):
                    series_index.append(index)

            # Initialize empty array
            src_array = np.array([]).reshape(len(series_index), source_data.shape[1], source_data.shape[2], 0)
            tgt_array = np.array([]).reshape(len(series_index), target_data.shape[1], target_data.shape[2], 0)

            for channel in range(source_data.shape[3]):
                src_series_ch_data = source_data[:, :, :, channel]
                tgt_series_ch_data = target_data[:, :, :, channel]
                src_ch_data = src_series_ch_data[series_index]
                tgt_ch_data = tgt_series_ch_data[series_index]

                src_array = np.concatenate((src_array, src_ch_data[:, :, :, np.newaxis]), axis=3)
                tgt_array = np.concatenate((tgt_array, tgt_ch_data[:, :, :, np.newaxis]), axis=3)

            source_data_dict = src_array
            target_data_dict = tgt_array

            # Cache the data
            pat_obj.source_data = source_data_dict
            pat_obj.target_data = target_data_dict
        else:
            # Initialize empty dictionaries
            index_dict, source_data_dict, target_data_dict = {}, {}, {}

            for series_date, series_data in source_data.items():

                # Read the 2nd channel (Lungs only)
                src = series_data[:, :, :, 1]

                # Initialize empty list
                series_index = []

                # Total number of pixels
                total_pixels = src.shape[1] * src.shape[2]

                for index, single_slice in enumerate(src):

                    # Number of pixels that are lungs
                    num_pixels = single_slice[single_slice != single_slice.min()]

                    if len(num_pixels) >= (percentage * total_pixels):
                        series_index.append(index)
                index_dict[series_date] = series_index

            for series_date, series_data in source_data.items():

                series_index = index_dict[series_date]
                tgt_series_data = target_data[series_date]

                # Initialize empty array
                src_array = np.array([]).reshape(len(series_index), series_data.shape[1], series_data.shape[2], 0)
                tgt_array = np.array([]).reshape(len(series_index), tgt_series_data.shape[1], tgt_series_data.shape[2],
                                                 0)

                for channel in range(series_data.shape[3]):
                    src_series_ch_data = series_data[:, :, :, channel]
                    tgt_series_ch_data = tgt_series_data[:, :, :, channel]
                    src_ch_data = src_series_ch_data[series_index]
                    tgt_ch_data = tgt_series_ch_data[series_index]

                    src_array = np.concatenate((src_array, src_ch_data[:, :, :, np.newaxis]), axis=3)
                    tgt_array = np.concatenate((tgt_array, tgt_ch_data[:, :, :, np.newaxis]), axis=3)

                source_data_dict[series_date] = src_array
                target_data_dict[series_date] = tgt_array

            # Cache the data
            pat_obj.source_data = source_data_dict
            pat_obj.target_data = target_data_dict
        return source_data_dict, target_data_dict

    def UKSH_remove_slices(self, pat_obj, percentage):
        """Removes the slices of UKSH source data and target data that are less than the percentage of the
         background.

        :param pat_obj: Object of UKSH class
        :param percentage: Percentage threshold
        :return: source_data_dict (UKSH), target_data_dict (UKSH)
        """
        # Load the source and target data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        print('Removing Lungless data in the second channel.')

        src = source_data[:, :, :, 1]

        # Initialize empty list
        series_index = []

        # Total number of pixels
        total_pixels = src.shape[1] * src.shape[2]

        for index, single_slice in enumerate(src):

            # Number of pixels that are lungs
            num_pixels = single_slice[single_slice != single_slice.min()]

            if len(num_pixels) >= (percentage * total_pixels):
                series_index.append(index)

        # Initialize empty array
        src_array = np.array([]).reshape(len(series_index), source_data.shape[1], source_data.shape[2], 0)
        tgt_array = np.array([]).reshape(len(series_index), target_data.shape[1], target_data.shape[2], 0)

        for channel in range(source_data.shape[3]):
            src_series_ch_data = source_data[:, :, :, channel]
            tgt_ch_data = target_data[:, :, :, channel]
            src_ch_data = src_series_ch_data[series_index]
            tgt_ch_data = tgt_ch_data[series_index]

            src_array = np.concatenate((src_array, src_ch_data[:, :, :, np.newaxis]), axis=3)
            tgt_array = np.concatenate((tgt_array, tgt_ch_data[:, :, :, np.newaxis]), axis=3)
        source_data_dict = src_array
        target_data_dict = tgt_array

        # Cache the data
        pat_obj.source_data = source_data_dict
        pat_obj.target_data = target_data_dict
        return source_data_dict, target_data_dict

    def LTRC_save(self, pat_obj):
        """Saves the LTRC CT data and label map..

        :param pat_obj: Object of LTRC class
        :return: None
        """
        # Initialize parameters
        affine = pat_obj.affine
        data_id = pat_obj.data_id
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.merge_channel is True:

            # Initialize empty dictionaries
            src_merged, tgt_merged = {}, {}

            for series_num, src_series in source_data.items():
                tgt_series = target_data[series_num]

                src_merged_series = src_series[:, :, :, 0] + src_series[:, :, :, 1]
                src_merged_series = src_merged_series[:, :, :, np.newaxis]

                tgt_merged_series = tgt_series[:, :, :, 0] + tgt_series[:, :, :, 1]
                tgt_merged_series = tgt_merged_series[:, :, :, np.newaxis]

                src_merged[series_num] = src_merged_series
                tgt_merged[series_num] = tgt_merged_series

            # Cache the data
            pat_obj.source_data = src_merged
            pat_obj.target_data = tgt_merged

        # Re-initialize the data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        for series_num, src in source_data.items():

            tgt = target_data[series_num]

            series_src_data = nib.Nifti1Pair(src, affine[series_num])
            series_tgt_map = nib.Nifti1Pair(tgt, affine[series_num])

            # Save the CT data and Label map
            if pat_obj.condition == 'Healthy':
                file_name = 'LTRC_Healthy_' + data_id + '_' + series_num + '_ct.nii'
                nib.save(series_tgt_map, op.join(self.save_dir_target, file_name))

                file_name = 'LTRC_Healthy_' + data_id + '_' + series_num + '_label_map.nii'
                nib.save(series_src_data, op.join(self.save_dir_source, file_name))

            elif pat_obj.condition == 'ARDS':
                file_name = 'LTRC_ARDS_' + data_id + '_' + series_num + '_ct.nii'
                nib.save(series_tgt_map, op.join(self.save_dir_target, file_name))

                file_name = 'LTRC_ARDS_' + data_id + '_' + series_num + '_label_map.nii'
                nib.save(series_src_data, op.join(self.save_dir_source, file_name))

        if self.LTRC_normalization.called and self.LTRC_resize.called:
            print('[LTRC] Normalized and Resized CT data and Label Map saved.\n')
        elif self.LTRC_normalization.called:
            print('[LTRC] Normalized CT data and Label Map saved.\n')
        pass

    def UMM_save(self, pat_obj):
        """Saves the UMM CT data and label map..

        :param pat_obj: Object of UMM class
        :return: None
        """
        # Initialize parameters
        affine = pat_obj.affine
        data_id = pat_obj.data_id
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if self.merge_channel is True:

            if not isinstance(source_data, dict):

                src_merged = source_data[:, :, :, 0] + source_data[:, :, :, 1]
                src_merged = src_merged[:, :, :, np.newaxis]

                tgt_merged = target_data[:, :, :, 0] + target_data[:, :, :, 1]
                tgt_merged = tgt_merged[:, :, :, np.newaxis]

                # Cache the data
                pat_obj.source_data = src_merged
                pat_obj.target_data = tgt_merged
            else:
                # Initialize empty dictionaries
                src_merged, tgt_merged = {}, {}

                for series_num, src_series in source_data.items():
                    tgt_series = target_data[series_num]

                    src_merged_series = src_series[:, :, :, 0] + src_series[:, :, :, 1]
                    src_merged_series = src_merged_series[:, :, :, np.newaxis]

                    tgt_merged_series = tgt_series[:, :, :, 0] + tgt_series[:, :, :, 1]
                    tgt_merged_series = tgt_merged_series[:, :, :, np.newaxis]

                    src_merged[series_num] = src_merged_series
                    tgt_merged[series_num] = tgt_merged_series

                # Cache the data
                pat_obj.source_data = src_merged
                pat_obj.target_data = tgt_merged

        # Re-initialize the data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        if not isinstance(source_data, dict):

            src_data = nib.Nifti1Pair(source_data, affine)
            tgt_data = nib.Nifti1Pair(target_data, affine)

            # Save the CT data and Label map
            file_name = 'UMM_ARDS_' + data_id + '_' + '2000_01_01' + '_ct.nii'
            nib.save(tgt_data, op.join(self.save_dir_target, file_name))

            file_name = 'UMM_ARDS_' + data_id + '_' + '2000_01_01' + '_label_map.nii'
            nib.save(src_data, op.join(self.save_dir_source, file_name))
        else:
            for series_date, src in source_data.items():
                tgt = target_data[series_date]

                series_src_data = nib.Nifti1Pair(src, affine[series_date])
                series_tgt_map = nib.Nifti1Pair(tgt, affine[series_date])

                # Save the CT data and Label map
                file_name = 'UMM_ARDS_' + data_id + '_' + series_date + '_ct.nii'
                nib.save(series_tgt_map, op.join(self.save_dir_target, file_name))

                file_name = 'UMM_ARDS_' + data_id + '_' + series_date + '_label_map.nii'
                nib.save(series_src_data, op.join(self.save_dir_source, file_name))

        if self.UMM_normalization.called and self.UMM_resize.called:
            print('[UMM] Normalized and Resized CT data and Label Map saved.\n')
        elif self.UMM_normalization.called:
            print('[UMM] Normalized CT data and Label Map saved.\n')
        pass

    def UKSH_save(self, pat_obj):
        """Saves the UKSH CT data and label map.

        :param pat_obj: Object of UKSH class
        :return: None
        """
        # Initialize parameters
        affine = pat_obj.affine
        data_id = pat_obj.data_id
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data
        series_date = pat_obj.series

        if self.merge_channel is True:
            src_merged = source_data[:, :, :, 0] + source_data[:, :, :, 1]
            src_merged = src_merged[:, :, :, np.newaxis]

            tgt_merged = target_data[:, :, :, 0] + target_data[:, :, :, 1]
            tgt_merged = tgt_merged[:, :, :, np.newaxis]

            pat_obj.source_data = src_merged
            pat_obj.target_data = tgt_merged

        # Re-initialize the data
        source_data = pat_obj.source_data
        target_data = pat_obj.target_data

        src_data = nib.Nifti1Pair(source_data, affine)
        tgt_data = nib.Nifti1Pair(target_data, affine)

        file_name = 'UKSH_ARDS_' + data_id + '_' + series_date + '_ct.nii'
        nib.save(tgt_data, op.join(self.save_dir_target, file_name))

        file_name = 'UKSH_ARDS_' + data_id + '_' + series_date + '_label_map.nii'
        nib.save(src_data, op.join(self.save_dir_source, file_name))

        if self.UKSH_normalization.called and self.UKSH_resize.called:
            print('[UKSH] Normalized and Resized CT data and Label Map saved.\n')
        elif self.UKSH_normalization.called:
            print('[UKSH] Normalized CT data and Label Map saved.\n')
        pass

    @calltracker
    def normalization(self, pat_obj):
        """Performs the normalization depending on the object.

        :param pat_obj: Object of either LTRC, UMM or UKSH class
        :return: None
        """
        if pat_obj.data_name == 'LTRC':
            _, _ = self.LTRC_normalization(pat_obj)
        elif pat_obj.data_name == 'UMM':
            _, _ = self.UMM_normalization(pat_obj)
        elif pat_obj.data_name == 'UKSH':
            _, _ = self.UKSH_normalization(pat_obj)
        pass

    @calltracker
    def resize(self, pat_obj, size):
        """Performs the resizing depending on the object.

        :param pat_obj: Object of either LTRC, UMM or UKSH class
        :param size: The size of the image to resize to
        :return: None
        """
        if pat_obj.data_name == 'LTRC':
            _, _ = self.LTRC_resize(pat_obj, size)
        elif pat_obj.data_name == 'UMM':
            _, _ = self.UMM_resize(pat_obj, size)
        elif pat_obj.data_name == 'UKSH':
            _, _ = self.UKSH_resize(pat_obj, size)
        pass

    def remove_slices(self, pat_obj, percentage=0.10):
        """Performs the resizing depending on the object.

        :param pat_obj: Object of either LTRC, UMM or UKSH class
        :param percentage: Percentage threshold
        :return: None
        """
        if pat_obj.data_name == 'LTRC':
            _, _ = self.LTRC_remove_slices(pat_obj, percentage)
        elif pat_obj.data_name == 'UMM':
            _, _ = self.UMM_remove_slices(pat_obj, percentage)
        elif pat_obj.data_name == 'UKSH':
            _, _ = self.UKSH_remove_slices(pat_obj, percentage)
        pass

    def save(self, pat_obj):
        """Saves the CT data and label map depending on the object.

        :param pat_obj: Object of either LTRC, UMM or UKSH class
        :return: None
        """
        assert self.normalization.called, 'Please perform normalization first.'

        if pat_obj.data_name == 'LTRC':
            self.LTRC_save(pat_obj)
        elif pat_obj.data_name == 'UMM':
            self.UMM_save(pat_obj)
        elif pat_obj.data_name == 'UKSH':
            self.UKSH_save(pat_obj)
        pass
