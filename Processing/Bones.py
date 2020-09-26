from Dataset import Dataset
from Processing import Lungs

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
from skimage.filters import sobel


class Bones:
    """Bone binary map for the respective datasets."""
    def __init__(self, pat_obj, bones_threshold=None):
        """Initialize the parameter for bone threshold value.

        :param pat_obj (object): Object of Dataset (LTRC, UMM or UKSH)
        :param bones_threshold (int): Threshold value for segmentation of bone
        """
        assert (isinstance(pat_obj, Dataset)), 'Object is not instance of Dataset'
        self.pat_obj = pat_obj
        self.pat_obj.bones_bin_map = None

        if bones_threshold is None:
            self.bones_threshold = 150
        else:
            self.bones_threshold = bones_threshold

    def __repr__(self):
        return '{self.__class__.__name__}(pat_obj={self.pat_obj}, bones_threshold={self.bones_threshold})' \
            .format(self=self)

    def LTRC_bone(self):
        """Segments the bone from single LTRC dataset.

        :return: bin_map (bone)
        """
        # Check if lungs binary map are available
        if hasattr(self.pat_obj, 'lungs_bin_map') is False:
            self.pat_obj.lungs_bin_map = Lungs(self.pat_obj, emphysema_va1=None, ventilated_val=None,
                                               poorly_vent_val=None, atelectatic_val=None)\
                .binary_map().pat_obj.lungs_bin_map

        # Initialize the parameters
        data_shape = self.pat_obj.data_shape
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map

        # Initialize empty dictionary
        bin_map = {}

        print('Creating Binary Map [LTRC]- Bones ' + '(Threshold value: ' + str(self.bones_threshold) + ')')
        # Create the bone binary map
        for series_num, ct_ in ori_ct.items():
            bones_edges_array = np.zeros(data_shape[series_num])
            ct = ct_.copy()

            # Remove the lungs from the CT images
            series_lungs_map = lungs_map[series_num]
            ct[series_lungs_map != 0] = ct.min()

            # Segmentation for the bones
            ct[ct < self.bones_threshold] = 0
            ct[ct != 0] = 1

            # Remove salt and pepper noise
            ct_filt = median_filter(ct, 5)

            # Fill empty areas with white pixels
            for i in range(ct_filt.shape[2]):
                bones_edges = sobel(ct_filt[:, :, i])
                bones_edges_array[:, :, i] = ndi.binary_fill_holes(bones_edges)

            bin_map[series_num] = bones_edges_array

        # Cache the data
        self.pat_obj.bones_bin_map = bin_map
        return bin_map

    def UMM_bone(self):
        """Segments the bone from single UMM dataset.

        :return: bin_map (bone)
        """
        # Check if lungs binary map are available
        if hasattr(self.pat_obj, 'lungs_bin_map') is False:
            self.pat_obj.lungs_bin_map = Lungs(self.pat_obj, emphysema_va1=None, ventilated_val=None,
                                               poorly_vent_val=None, atelectatic_val=None)\
                .binary_map().pat_obj.lungs_bin_map

        # Initialize the parameters and array
        data_shape = self.pat_obj.data_shape
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map

        print('Creating Binary Map [UMM]- Bones ' + '(Threshold value: ' + str(self.bones_threshold) + ')')

        if not isinstance(ori_ct, dict):
            bones_edges_array = np.zeros(data_shape)

            ct = ori_ct.copy()

            # Remove the lungs from the CT images
            ct[lungs_map != 0] = ct.min()

            # Segmentation for the bones
            ct[ct < self.bones_threshold] = 0
            ct[ct != 0] = 1

            # Remove salt and pepper noise
            ct_filt = median_filter(ct, 5)

            # Fill empty areas with white pixels
            for i in range(ct_filt.shape[2]):
                bones_edges = sobel(ct_filt[:, :, i])
                bones_edges_array[:, :, i] = ndi.binary_fill_holes(bones_edges)

            # Cache the data
            bin_map = bones_edges_array
            self.pat_obj.bones_bin_map = bones_edges_array
        else:
            bin_map = {}

            for series_date, ct_ in ori_ct.items():
                bones_edges_array = np.zeros(data_shape[series_date])

                ct = ct_.copy()

                # Remove the lungs from the CT images
                series_lungs_map = lungs_map[series_date]
                ct[series_lungs_map != 0] = ct.min()

                # Segmentation for the bones
                ct[ct < self.bones_threshold] = 0
                ct[ct != 0] = 1

                # Remove salt and pepper noise
                ct_filt = median_filter(ct, 5)

                # Fill empty areas with white pixels
                for i in range(ct_filt.shape[2]):
                    bones_edges = sobel(ct_filt[:, :, i])
                    bones_edges_array[:, :, i] = ndi.binary_fill_holes(bones_edges)

                bin_map[series_date] = bones_edges_array

        # Cache the data
        self.pat_obj.bones_bin_map = bin_map
        return bin_map

    def UKSH_bone(self):
        """Segments the bone from single UKSH dataset.

        :return: bin_map (bone)
        """
        # Check if lungs binary map are available
        if hasattr(self.pat_obj, 'lungs_bin_map') is False:
            self.pat_obj.lungs_bin_map = Lungs(self.pat_obj, emphysema_va1=None, ventilated_val=None,
                                               poorly_vent_val=None, atelectatic_val=None)\
                .binary_map().pat_obj.lungs_bin_map

        # Initialize the parameters and array
        data_shape = self.pat_obj.data_shape
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map
        bones_edges_array = np.zeros(data_shape)

        print('Creating Binary Map [UKSH]- Bones ' + '(Threshold value: ' + str(self.bones_threshold) + ')')
        ct = ori_ct.copy()

        # Remove the lungs from the CT images
        ct[lungs_map != 0] = ct.min()

        # Segmentation for the bones
        ct[ct < self.bones_threshold] = 0
        ct[ct != 0] = 1

        # Remove salt and pepper noise
        ct_filt = median_filter(ct, 5)

        # Fill empty areas with white pixels
        for i in range(ct_filt.shape[2]):
            bones_edges = sobel(ct_filt[:, :, i])
            bones_edges_array[:, :, i] = ndi.binary_fill_holes(bones_edges)

        # Cache the data
        bin_map = bones_edges_array
        self.pat_obj.bones_bin_map = bin_map
        return bin_map

    def binary_map(self):
        """Performs the bone segmentation depending on the object.

        :return: self
        """
        if self.pat_obj.data_name == 'LTRC':
            _ = self.LTRC_bone()
        elif self.pat_obj.data_name == 'UMM':
            _ = self.UMM_bone()
        elif self.pat_obj.data_name == 'UKSH':
            _ = self.UKSH_bone()
        return self
