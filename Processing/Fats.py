from Dataset import Dataset
from Processing import Tissues

from scipy.ndimage import median_filter


class Fats:
    """Fats binary map for the respective datasets."""
    def __init__(self, pat_obj, fats_threshold=None):
        """Initialize the parameter for tissues threshold value.

        :param pat_obj (object): Object of Dataset (LTRC, UMM or UKSH)
        :param fats_threshold (int): Threshold value for segmentation of fats
        """
        assert (isinstance(pat_obj, Dataset)), 'Object is not instance of Dataset'
        self.pat_obj = pat_obj
        self.pat_obj.fats_bin_map = None

        if fats_threshold is None:
            self.fats_threshold = -400
        else:
            self.fats_threshold = fats_threshold

    def __repr__(self):
        return '{self.__class__.__name__}(pat_obj={self.pat_obj}, fats_threshold={self.fats_threshold})'\
            .format(self=self)

    def LTRC_fats(self):
        """Segments the fats from single LTRC dataset.

        :return: bin_map (fats)
        """
        # Check if tissues binary map are available
        if hasattr(self.pat_obj, 'tissues_bin_map') is False:
            self.pat_obj.tissues_bin_map = Tissues(self.pat_obj).binary_map().pat_obj.tissues_bin_map

        # Initialize empty dictionary and parameter
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map
        bones_data = self.pat_obj.bones_bin_map
        tissues_data = self.pat_obj.tissues_bin_map
        bin_map = {}

        print('Creating Binary Map [LTRC]- Fats ' + '(Threshold value: ' + str(self.fats_threshold) + ')')
        # Read single series
        for series_num, ct_ in ori_ct.items():

            ct = ct_.copy()

            # Remove the lungs, bones and tissues from the CT images
            series_lungs_map = lungs_map[series_num]
            series_bones_data = bones_data[series_num]
            series_tissues_data = tissues_data[series_num]
            ct[series_lungs_map != 0] = ct.min()
            ct[series_bones_data != 0] = ct.min()
            ct[series_tissues_data != 0] = ct.min()

            # Segmentation for the fats
            ct[ct <= self.fats_threshold] = 0
            ct[ct != 0] = 1

            # Remove salt and pepper noise
            ct_filt = median_filter(ct, 3)

            bin_map[series_num] = ct_filt

        # Cache the data
        self.pat_obj.fats_bin_map = bin_map
        return bin_map

    def UMM_fats(self):
        """Segments the fats from single UMM dataset.

        :return: bin_map (fats)
        """
        # Check if tissues binary map are available
        if hasattr(self.pat_obj, 'tissues_bin_map') is False:
            self.pat_obj.tissues_bin_map = Tissues(self.pat_obj).binary_map().pat_obj.tissues_bin_map

        # Initialize empty dictionary and parameter
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map
        bones_data = self.pat_obj.bones_bin_map
        tissues_data = self.pat_obj.tissues_bin_map

        print('Creating Binary Map [UMM]- Fats ' + '(Threshold value: ' + str(self.fats_threshold) + ')')

        if not isinstance(ori_ct, dict):

            ct = ori_ct.copy()

            # Remove the lung, bones, tissues from the CT images
            ct[lungs_map != 0] = ct.min()
            ct[bones_data != 0] = ct.min()
            ct[tissues_data != 0] = ct.min()

            # Segmentation for the fats
            ct[ct <= self.fats_threshold] = 0
            ct[ct != 0] = 1

            # Remove salt and pepper noise
            bin_map = median_filter(ct, 3)
        else:
            bin_map = {}

            for series_date, ct_ in ori_ct.items():

                ct = ct_.copy()

                single_lung_map = lungs_map[series_date]
                single_bone_data = bones_data[series_date]
                single_tissue_data = tissues_data[series_date]

                # Remove the lung, bones, tissues from the CT images
                ct[single_lung_map != 0] = -1024
                ct[single_bone_data != 0] = -1024
                ct[single_tissue_data != 0] = -1024

                # Segmentation for the fats
                ct[ct < self.fats_threshold] = 0
                ct[ct != 0] = 1

                # Remove salt and pepper noise
                ct_filt = median_filter(ct, 3)

                bin_map[series_date] = ct_filt

        # Cache the data
        self.pat_obj.fats_bin_map = bin_map
        return bin_map

    def UKSH_fats(self):
        """Segments the fats from single UKSH dataset.

        :return: bin_map (fats)
        """
        # Check if tissues binary map are available
        if hasattr(self.pat_obj, 'tissues_bin_map') is False:
            self.pat_obj.tissues_bin_map = Tissues(self.pat_obj).binary_map().pat_obj.tissues_bin_map

        # Initialize the parameters and array
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_bin_map
        bones_data = self.pat_obj.bones_bin_map
        tissues_data = self.pat_obj.tissues_bin_map

        print('Creating Binary Map [UKSH]- Fats ' + '(Threshold value: ' + str(self.fats_threshold) + ')')
        ct = ori_ct.copy()

        # Remove the lung, bones, tissues from the CT images
        ct[lungs_map != 0] = ct.min()
        ct[bones_data != 0] = ct.min()
        ct[tissues_data != 0] = ct.min()

        # Segmentation for the fats
        ct[ct < self.fats_threshold] = 0
        ct[ct != 0] = 1

        # Remove salt and pepper noise
        bin_map = median_filter(ct, 3)

        # Cache the data
        self.pat_obj.fats_bin_map = bin_map
        return bin_map

    def binary_map(self):
        """Performs the fats segmentation depending on the object.

        :return: self
        """
        if self.pat_obj.data_name == 'LTRC':
            _ = self.LTRC_fats()
        elif self.pat_obj.data_name == 'UMM':
            _ = self.UMM_fats()
        elif self.pat_obj.data_name == 'UKSH':
            _ = self.UKSH_fats()
        return self
