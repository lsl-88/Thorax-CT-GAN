from Dataset import Dataset

import os
import os.path as op
from copy import copy
import nibabel as nib


class LTRC(Dataset):
    """Single LTRC (Healthy) dataset consists of CT images and its corresponding binary maps."""
    data_type = 'CT'
    data_name = 'LTRC'
    condition = 'Healthy'
    _cache_list = []

    def __init__(self, data_id, data_dir=None, **kwargs):
        """Initialize a LTRC (Healthy) dataset without loading it.

        :param data_id: String identifier for the dataset, e.g. '102022'
        :param data_dir: The path to the data directory in which the LTRC dataset resides
        :param **kwargs: Arbitrary keyword arguments (unused)
        """
        super().__init__(**kwargs)

        if data_dir is None:
            if data_dir.split('/')[-1] != self.__class__.data_name:
                self.data_dir = op.join('/home/ubuntu/sl_root/Data/', self.__class__.data_name)
            else:
                self.data_dir = '/home/ubuntu/sl_root/Data/LTRC'
        else:
            if data_dir.split('/')[-1] != self.__class__.data_name:
                self.data_dir = op.join(data_dir, self.__class__.data_name)
            else:
                self.data_dir = data_dir
        self.data_id = data_id
        self.cache(data_id)

        # Check LTRC dataset
        assert op.exists(op.join(self.data_dir, self.data_id)), 'Patient-{data_id} data unavailable' \
            .format(data_id=self.data_id)

        # Variables to store data
        self.ct_data = None
        self.seg_data = None
        self.series = None
        self.scan_type = []
        self.data_shape = None
        self.affine = None

    def __repr__(self):
        return '{self.__class__.__name__}(data_id={self.data_id}, data_dir={self.data_dir}, **kwargs)' \
            .format(self=self)

    def cache(self, data_id):
        if data_id not in self._cache_list:
            self._cache_list.append(data_id)
        return self

    def instances(self):
        return len(self._cache_list)

    def load(self, data_type='all', verbose=True):
        """Loads a series dataset.

        :param data_type: Data type to load ('CT', 'seg' or 'all')
        :param verbose: Print out the progress of loading
        :return: Instance to the dataset (i.e. 'self')
        """
        assert data_type == 'CT' or data_type == 'seg' or data_type == 'all', 'Please select the appropriate data type'

        # Load CT data only
        if data_type == 'CT':
            if verbose:
                print('\nLoading: ' + str(self.data_name) + '-' + str(self.data_id) + ' - CT data')
            self.load_ct(self.data_id)

        # Load Seg data only
        elif data_type == 'seg':
            if verbose:
                print('\nLoading: ' + str(self.data_name) + '-' + str(self.data_id) + ' - segmentation data')
            self.load_seg(self.data_id)

        # Load both CT and Seg data
        elif data_type == 'all':
            if verbose:
                print('\nLoading: ' + str(self.data_name) + '-' + str(self.data_id))
            self.load_ct(self.data_id)
            self.load_seg(self.data_id)
        return self

    def load_ct(self, data_id):
        """Loads the CT data of the patient.

        :param data_id: String identifier for the dataset, e.g. '102022'
        :return: Instance to the dataset (i.e. 'self')
        """
        # Set the dataset directory
        dataset_dir = op.join(self.data_dir, data_id)

        # Initialize empty list and dictionary
        series_ct_data = {}
        series_affine = {}
        series_data_shape = {}
        series_list, scan_type_list = [], []

        for single_series in os.listdir(dataset_dir):

            series_num = single_series.split('_')[2] + '_' + single_series.split('_')[3]
            series_list.append(series_num)

            # Set the CT directory
            ct_dir = op.join(dataset_dir, single_series, 'nifti')

            for single_data in os.listdir(ct_dir):
                scan_type = single_data.split('_')[4].split('.')[0]
                assert scan_type == self.data_type, 'The data is not CT data.'
                # Append the scan_type to list
                scan_type_list.append(scan_type)

                # Load the data of the organ
                img = nib.load(op.join(ct_dir, single_data))

                # Append to the dictionary
                series_ct_data[series_num] = img.get_fdata()
                series_affine[series_num] = img.affine
                series_data_shape[series_num] = img.get_fdata().shape

        # Cache the data
        self.ct_data = series_ct_data
        self.data_shape = series_data_shape
        self.series = list(set(series_list))
        self.scan_type = list(set(scan_type_list))
        self.affine = series_affine
        return self

    def load_seg(self, data_id):
        """Loads the segmentation data of the patient.

        :param data_id: String identifier for the dataset, e.g. '102022'
        :return: Instance to the dataset (i.e. 'self')
        """
        # Set the dataset directory
        dataset_dir = op.join(self.data_dir, data_id)

        # Initialize empty list and dictionary
        organs_data, series_seg_data = {}, {}
        series_list, scan_type_list = [], []

        for single_series in os.listdir(dataset_dir):

            series_num = single_series.split('_')[2] + '_' + single_series.split('_')[3]
            series_list.append(series_num)

            # Set the segmentation directory
            seg_dir = op.join(dataset_dir, single_series, 'segmentation')

            for single_data in os.listdir(seg_dir):

                scan_type = single_data.split('_')[4].split('.')[0]

                if scan_type == 'lungs' or scan_type == 'airways' or scan_type == 'vessels':
                    # Append the scan_type to list
                    scan_type_list.append(scan_type)

                    # Load the data of the organ
                    img = nib.load(op.join(seg_dir, single_data))

                    # Append to the dictionary
                    organs_data[scan_type] = img.get_fdata()

                # Need to copy the series data
                organs_data_copy = copy(organs_data)

            # Append to series segmentation data
            series_seg_data[series_num] = organs_data_copy

        # Cache the data
        self.seg_data = series_seg_data
        self.series = list(set(series_list))
        self.scan_type = list(set(scan_type_list))
        return self
