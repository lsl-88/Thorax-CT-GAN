from Dataset import Dataset
import numpy as np


class LunglessCT:
    """Original CT without lungs for the respective datasets."""
    def __init__(self, pat_obj):
        """Initialize the parameter for lungs threshold value.

        :param pat_obj: Object of Dataset (LTRC, UMM or UKSH)
        """
        assert (isinstance(pat_obj, Dataset)), 'Object is not instance of Dataset'
        self.pat_obj = pat_obj
        self.pat_obj.lungless_ct = None

    def __repr__(self):
        return '{self.__class__.__name__}(pat_obj={self.pat_obj})'.format(self=self)

    def LTRC_lungs(self):
        """Segments the lungs out of the CT data from single LTRC dataset.

        :return: bin_map (lungless CT)
        """
        # Initialize empty dictionary and parameter
        ori_ct = self.pat_obj.ct_data
        seg_data = self.pat_obj.seg_data
        bin_map = {}

        print('Creating Lungless CT [LTRC]')

        # Read single series
        for series_num, ct_ in ori_ct.items():

            lungs_ct = ct_.copy()
            single_seg_data = seg_data[series_num]

            # Remove the lungs from the CT images
            lungs_map = np.zeros(self.pat_obj.data_shape[series_num])
            for _, org_data in single_seg_data.items():
                lungs_map = lungs_map + org_data
            lungs_ct[lungs_map != 0] = lungs_ct.min()

            bin_map[series_num] = lungs_ct

        # Cache the data
        self.pat_obj.lungless_ct = bin_map
        return bin_map

    def UMM_lungs(self):
        """Segments the lungs out of the CT data from single UMM dataset.

        :return: bin_map (lungless CT)
        """
        # Initialize the parameters and array
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_data

        print('Creating Lungless CT [UMM]')

        if not isinstance(ori_ct, dict):

            lungs_ct = ori_ct.copy()

            # Remove the lungs from the CT images
            lungs_ct[lungs_map != 0] = lungs_ct.min()

            # Cache the data
            self.pat_obj.lungless_ct = lungs_ct
            bin_map = lungs_ct

        else:
            bin_map = {}

            for series_date, ct_ in ori_ct.items():

                lungs_ct = ct_.copy()

                # Remove the lungs from the CT images
                lungs_ct[lungs_map[series_date] != 0] = lungs_ct.min()

                bin_map[series_date] = lungs_ct

            # Cache the data
            self.pat_obj.lungless_ct = bin_map
        return bin_map

    def UKSH_lungs(self):
        """Segments the lungs out of the CT data from single UKSH dataset.

        :return: bin_map (lungless CT)
        """
        # Initialize the parameters and array
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_data

        print('Creating Lungless CT [UKSH]')

        lungs_ct = ori_ct.copy()

        # Remove the lungs from the CT images
        lungs_ct[lungs_map != 0] = lungs_ct.min()

        # Cache the data
        self.pat_obj.lungless_ct = lungs_ct
        bin_map = lungs_ct
        return bin_map

    def binary_map(self):
        """Creates the lungless CT data depending on the object.

        :return: self
        """
        if self.pat_obj.data_name == 'LTRC':
            _ = self.LTRC_lungs()
        elif self.pat_obj.data_name == 'UMM':
            _ = self.UMM_lungs()
        elif self.pat_obj.data_name == 'UKSH':
            _ = self.UKSH_lungs()
        return self

