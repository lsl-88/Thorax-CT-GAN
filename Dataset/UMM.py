from Dataset import Dataset

import os
import os.path as op
import nibabel as nib


class UMM(Dataset):
    """Single UMM dataset consists of CT images and its corresponding binary maps."""
    data_type = 'CT'
    data_name = 'UMM'
    _cache_list = []

    def __init__(self, data_id, data_dir=None, **kwargs):
        """Initialize a UMM dataset without loading it.

        :param data_id: String identifier for the dataset, e.g. 'pat0001'
        :param data_dir: The path to the data directory in which the UMM dataset resides
        :param **kwargs: Arbitrary keyword arguments (unused)
        """
        super().__init__(**kwargs)

        if data_dir is None:
            if data_dir.split('/')[-1] != self.__class__.data_name:
                self.data_dir = op.join('/home/ubuntu/sl_root/Data/', self.__class__.data_name)
            else:
                self.data_dir = '/home/ubuntu/sl_root/Data/UMM'
        else:
            if data_dir.split('/')[-1] != self.__class__.data_name:
                self.data_dir = op.join(data_dir, self.__class__.data_name)
            else:
                self.data_dir = data_dir
        self.data_id = data_id
        self.cache(data_id)

        # Check UMM dataset
        assert op.exists(op.join(self.data_dir, self.data_id)), 'Patient-{data_id} data unavailable'.format(
            data_id=self.data_id)

        # Variables to store data
        self.ct_data = None
        self.lungs_data = None
        self.lungs_areas_data = None
        self.data_shape = None
        self.affine = None
        self.series = None

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
        :param verbose: Print out the progress of loading.
        :return: Instance to the dataset (i.e. 'self')
        """
        assert data_type == 'CT' or data_type == 'seg' or data_type == 'all', 'Please select the appropriate data type'

        # Load CT data only
        if data_type == 'CT':
            if verbose:
                print('Loading: ' + str(self.data_name) + '-' + str(self.data_id) + ' - CT data')
            self.load_ct(self.data_id)

        # Load Seg data only
        elif data_type == 'seg':
            if verbose:
                print('Loading: ' + str(self.data_name) + '-' + str(self.data_id) + ' - segmentation data')
            self.load_seg(self.data_id)

        # Load both CT and Seg data
        elif data_type == 'all':
            if verbose:
                print('Loading: ' + str(self.data_name) + '-' + str(self.data_id))
            self.load_ct(self.data_id)
            self.load_seg(self.data_id)
        return self

    def load_ct(self, data_id):
        """Loads the CT data of the patient.

        :param data_id: String identifier for the dataset, e.g. 'pat0001'
        :return: Instance to the dataset (i.e. 'self')
        """
        # Set the dataset directory
        dataset_dir = op.join(self.data_dir, data_id)

        # Initialize empty list and dictionary
        series_ct_data, series_data_shape, series_affine = {}, {}, {}
        series_list = []

        for file_type in os.listdir(dataset_dir):
            if file_type == 'thx_endex':

                pat_dataset = op.join(dataset_dir, file_type, 'nifti')
                for ct in os.listdir(pat_dataset):
                    if ct.split('_')[0] == data_id:
                        img = nib.load(op.join(pat_dataset, ct))
                        ct_data = img.get_fdata()
                        affine = img.affine

                self.ct_data = ct_data
                self.data_shape = ct_data.shape
                self.series = '2000_01_01'
                self.affine = affine

            elif file_type.split('_')[1] == 'endex':
                year = file_type.split('_')[2]
                month = file_type.split('_')[3]
                day = file_type.split('_')[4]
                date = year + '_' + month + '_' + day
                series_list.append(date)

                pat_dataset = op.join(dataset_dir, 'thx_endex_' + date, 'nifti')
                for ct in os.listdir(pat_dataset):
                    if ct.split('_')[0] == data_id:
                        img = nib.load(op.join(pat_dataset, ct))
                        series_ct_data[date] = img.get_fdata()
                        series_data_shape[date] = img.get_fdata().shape
                        series_affine[date] = img.affine

                # Cache the data
                self.ct_data = series_ct_data
                self.data_shape = series_data_shape
                self.series = list(set(series_list))
                self.affine = series_affine
        return self

    def load_seg(self, data_id):
        """Loads the segmentation data of the patient.

        :param data_id: String identifier for the dataset, e.g. 'pat0001'
        :return: Instance to the dataset (i.e. 'self')
        """
        # Set the dataset directory
        dataset_dir = op.join(self.data_dir, data_id)

        # Initialize empty dictionaries
        series_lungs_data, series_lungs_areas_data, series_data_shape, series_affine = {}, {}, {}, {}

        for file_type in os.listdir(dataset_dir):
            if file_type == 'thx_endex':

                pat_dataset = op.join(dataset_dir, file_type, 'segmentation')
                for single_file in os.listdir(pat_dataset):
                    if single_file.split('.')[1] == 'nii':
                        if single_file.split('.nii')[0].split('_')[-1] == 'lung':
                            img_lungs = nib.load(op.join(pat_dataset, single_file))
                            self.lungs_data = img_lungs.get_fdata()
                        else:
                            img_lungs_areas = nib.load(op.join(pat_dataset, single_file))
                            self.lungs_areas_data = img_lungs_areas.get_fdata()

            elif file_type.split('_')[1] == 'endex':
                year = file_type.split('_')[2]
                month = file_type.split('_')[3]
                day = file_type.split('_')[4]
                date = year + '_' + month + '_' + day

                pat_dataset = op.join(dataset_dir, 'thx_endex_' + date, 'segmentation')
                for single_file in os.listdir(pat_dataset):
                    if single_file.split('.')[1] == 'nii':
                        if single_file.split('.nii')[0].split('_')[-1] == 'lung':
                            img_lungs = nib.load(op.join(pat_dataset, single_file))
                            series_lungs_data[date] = img_lungs.get_fdata()
                            series_data_shape[date] = img_lungs.get_fdata().shape
                            series_affine[date] = img_lungs.affine

                            self.lungs_data = series_lungs_data
                        else:
                            img_lungs_areas = nib.load(op.join(pat_dataset, single_file))
                            series_lungs_areas_data[date] = img_lungs_areas.get_fdata()

                            self.lungs_areas_data = series_lungs_areas_data
        return self
