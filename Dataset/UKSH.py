from Dataset import Dataset

import os
import os.path as op
import nibabel as nib


class UKSH(Dataset):
    """Single UKSH dataset consists of CT images and its corresponding binary maps."""
    data_type = 'CT'
    data_name = 'UKSH'
    _cache_list = []

    def __init__(self, data_id, data_dir=None, **kwargs):
        """Initialize a UKSH dataset without loading it.

        :param data_id (str): String identifier for the dataset, e.g. 'A10'
        :param data_dir (str): The path to the data directory in which the UKSH dataset resides
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

        # Check UKSH dataset
        assert op.exists(op.join(self.data_dir, self.data_id)), 'Patient-{data_id} data unavailable' \
            .format(data_id=self.data_id)

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
        :param verbose : Print out the progress of loading
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

        :param data_id: String identifier for the dataset, e.g. 'A10'
        """
        # Set the dataset directory
        pat_dataset = op.join(self.data_dir, data_id, 'nifti')

        for file in os.listdir(pat_dataset):
            if file[:4] == 'UKSH':
                img = nib.load(op.join(pat_dataset, file))
                ct_data = img.get_fdata()
                affine = img.affine

        # Cache the data
        self.ct_data = ct_data
        self.data_shape = ct_data.shape
        self.series = '2000_01_01'
        self.affine = affine
        return self

    def load_seg(self, data_id):
        """Loads the segmentation data of the patient.

        :param data_id: String identifier for the dataset, e.g. 'A10'
        :return: Instance to the dataset (i.e. 'self')
        """
        # Set the dataset directory
        pat_dataset = op.join(self.data_dir, data_id, 'segmentation')

        for file in os.listdir(pat_dataset):
            if file[:4] == 'UKSH' and file.split('.')[0].split('_')[-1] == 'seg':
                img_lungs = nib.load(op.join(pat_dataset, file))
                self.lungs_data = img_lungs.get_fdata()

            elif file.split('_')[0][:4] == 'UKSH' and file.split('.')[0].split('_')[-1] == 'areas':
                img_lungs_areas = nib.load(op.join(pat_dataset, file))
                self.lungs_areas_data = img_lungs_areas.get_fdata()
        return self
