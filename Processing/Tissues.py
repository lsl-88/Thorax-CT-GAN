from Dataset import Dataset
from Processing import Bones

from scipy.ndimage import median_filter


class Tissues:
    """Tissues binary map for the respective datasets."""
    def __init__(self, pat_obj, tissues_threshold=None):
        """Initialize the parameter for tissues threshold value.

        :param pat_obj (object): Object of Dataset (LTRC, UMM or UKSH)
        :param tissues_threshold (int): Threshold value for segmentation of tissues
        """
        assert (isinstance(pat_obj, Dataset)), 'Object is not instance of Dataset'
        self.pat_obj = pat_obj
        self.pat_obj.tissues_bin_map = None

        if tissues_threshold is None:
            self.tissues_threshold = 0
        else:
            self.tissues_threshold = tissues_threshold

    def __repr__(self):
        return '{self.__class__.__name__}(pat_obj={self.pat_obj}, tissues_threshold={self.tissues_threshold})'\
            .format(self=self)

    def LTRC_tissues(self):
        """Segments the tissues from single LTRC dataset.

        :return: bin_map (tissues)
        """
        # Check if bone binary map are available
        if hasattr(self.pat_obj, 'bones_bin_map') is False:
            self.pat_obj.bones_bin_map = Bones(self.pat_obj).binary_map().pat_obj.bones_bin_map

        # Initialize empty dictionary and parameter
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map
        bones_data = self.pat_obj.bones_bin_map
        bin_map = {}

        print('Creating Binary Map [LTRC]- Tissues ' + '(Threshold value: ' + str(self.tissues_threshold) + ')')
        # Read single series
        for series_num, ct_ in ori_ct.items():

            ct = ct_.copy()

            # Remove the lungs and bones from the CT images
            series_lungs_map = lungs_map[series_num]
            series_bones_data = bones_data[series_num]
            ct[series_lungs_map != 0] = ct.min()
            ct[series_bones_data != 0] = ct.min()

            # Segmentation for the tissues
            ct[ct <= self.tissues_threshold] = 0
            ct[ct > self.tissues_threshold] = 1

            # Remove salt and pepper noise
            ct_filt = median_filter(ct, 7)

            bin_map[series_num] = ct_filt

        # Cache the data
        self.pat_obj.tissues_bin_map = bin_map
        return bin_map

    def UMM_tissues(self):
        """Segments the tissues from single UMM dataset.

        :return: bin_map (tissues)
        """
        # Check if bone binary map are available
        if hasattr(self.pat_obj, 'bones_bin_map') is False:
            self.pat_obj.bones_bin_map = Bones(self.pat_obj).binary_map().pat_obj.bones_bin_map

        # Initialize empty dictionary and parameter
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map
        bones_data = self.pat_obj.bones_bin_map

        print('Creating Binary Map [UMM]- Tissues ' + '(Threshold value: ' + str(self.tissues_threshold) + ')')

        if not isinstance(ori_ct, dict):

            ct = ori_ct.copy()

            # Remove the lung and bones from the CT images
            ct[lungs_map != 0] = ct.min()
            ct[bones_data != 0] = ct.min()

            # Segmentation for the tissues
            ct[ct <= self.tissues_threshold] = 0
            ct[ct != 0] = 1

            # Remove salt and pepper noise
            bin_map = median_filter(ct, 7)
        else:
            bin_map = {}

            for series_date, ct_ in ori_ct.items():

                ct = ct_.copy()

                # Remove the lung and bones from the CT images
                series_lung_data = lungs_map[series_date]
                series_bone_data = bones_data[series_date]
                ct[series_lung_data != 0] = ct.min()
                ct[series_bone_data != 0] = ct.min()

                # Segmentation for the tissues
                ct[ct < self.tissues_threshold] = 0
                ct[ct != 0] = 1

                # Remove salt and pepper noise
                ct_filt = median_filter(ct, 7)

                bin_map[series_date] = ct_filt

        # Cache the data
        self.pat_obj.tissues_bin_map = bin_map
        return bin_map

    def UKSH_tissues(self):
        """Segments the tissues from single UKSH dataset.

        :return: bin_map (tissues)
        """
        # Check if bone binary map are available
        if hasattr(self.pat_obj, 'bones_bin_map') is False:
            self.pat_obj.bones_bin_map = Bones(self.pat_obj).binary_map().pat_obj.bones_bin_map

        # Initialize the parameters and array
        ori_ct = self.pat_obj.ct_data
        lung_map = self.pat_obj.lungs_bin_map
        bones_data = self.pat_obj.bones_bin_map

        print('Creating Binary Map [UKSH]- Bones ' + '(Threshold value: ' + str(self.tissues_threshold) + ')')
        ct = ori_ct.copy()

        # Remove the lung and bones from the CT images
        ct[lung_map != 0] = ct.min()
        ct[bones_data != 0] = ct.min()

        # Segmentation for the tissues
        ct[ct < self.tissues_threshold] = 0
        ct[ct != 0] = 1

        # Remove salt and pepper noise
        bin_map = median_filter(ct, 7)

        # Cache the data
        self.pat_obj.tissues_bin_map = bin_map
        return bin_map

    def binary_map(self):
        """Performs the tissues segmentation depending on the object.

        :return: self
        """
        if self.pat_obj.data_name == 'LTRC':
            _ = self.LTRC_tissues()
        elif self.pat_obj.data_name == 'UMM':
            _ = self.UMM_tissues()
        elif self.pat_obj.data_name == 'UKSH':
            _ = self.UKSH_tissues()
        return self
