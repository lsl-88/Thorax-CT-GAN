from Dataset import Dataset
import numpy as np


class Lungs:
    """Lungs binary map for the respective datasets."""
    def __init__(self, pat_obj, emphysema_va1=None, ventilated_val=None, poorly_vent_val=None, atelectatic_val=None):
        """Initialize the parameter for lungs threshold value.

        :param pat_obj: Object of Dataset (LTRC, UMM or UKSH)
        :param emphysema_va1: Threshold value for emphysema
        :param ventilated_val: Threshold value for ventilated lungs
        :param poorly_vent_val: Threshold value for poorly ventilated lungs
        :param atelectatic_val: Threshold value for atelectatic lungs
        """
        assert (isinstance(pat_obj, Dataset)), 'Object is not instance of Dataset'
        self.pat_obj = pat_obj
        self.pat_obj.lungs_bin_map = None
        self.pat_obj.lungs_ct_data = None

        # Initialize the parameters for lungs segmentation
        self.emphysema_range = (-1024, -900)
        self.ventilated_range = (-900, -500)
        self.poorly_vent_range = (-500, -100)
        self.atelectatic_range = (-100, 100)

        # Initialize default parameters for lungs label map
        self._emphysema_val = 50
        self._ventilated_val = 300
        self._poorly_vent_val = 700
        self._atelectatic_val = 1000

        # Initialize the parameters
        if emphysema_va1 is None:
            self.emphysema_va1 = self._emphysema_val
        else:
            self.emphysema_va1 = emphysema_va1
        if ventilated_val is None:
            self.ventilated_val = self._ventilated_val
        else:
            self.ventilated_val = ventilated_val
        if poorly_vent_val is None:
            self.poorly_vent_val = self._poorly_vent_val
        else:
            self.poorly_vent_val = poorly_vent_val
        if atelectatic_val is None:
            self.atelectatic_val = self._atelectatic_val
        else:
            self.atelectatic_val = atelectatic_val

    def __repr__(self):
        return '{self.__class__.__name__}(pat_obj={self.pat_obj}, emphysema_va1={self.emphysema_va1}, ' \
                'ventilated_val={self.ventilated_val}, poorly_vent_val={self.poorly_vent_val}, ' \
                'atelectatic_val={self.atelectatic_val})'.format(self=self)

    def lungs_binary_map(self, series_lungs_ct):
        """Generates the features of ARDS from the segmented lungs.

        :return: emphysema, ventilated, poorly_vent, atelectatic
        """
        # Copy 4 sets for different conditions
        emphysema = series_lungs_ct.copy()
        ventilated = series_lungs_ct.copy()
        poorly_vent = series_lungs_ct.copy()
        atelectatic = series_lungs_ct.copy()

        # Segmentation for emphysema
        emphysema[emphysema < self.emphysema_range[0]] = emphysema.min()
        emphysema[emphysema > self.emphysema_range[1]] = emphysema.min()
        emphysema[emphysema != emphysema.min()] = 1
        emphysema[emphysema == emphysema.min()] = 0

        # Segmentation for ventilated lungs
        ventilated[ventilated < self.ventilated_range[0]] = ventilated.min()
        ventilated[ventilated > self.ventilated_range[1]] = ventilated.min()
        ventilated[ventilated != ventilated.min()] = 1
        ventilated[ventilated == ventilated.min()] = 0

        # Segmentation for poorly ventilated lungs
        poorly_vent[poorly_vent < self.poorly_vent_range[0]] = poorly_vent.min()
        poorly_vent[poorly_vent > self.poorly_vent_range[1]] = poorly_vent.min()
        poorly_vent[poorly_vent != poorly_vent.min()] = 1
        poorly_vent[poorly_vent == poorly_vent.min()] = 0

        # Segmentation for atelectatic
        atelectatic[atelectatic < self.atelectatic_range[0]] = atelectatic.min()
        atelectatic[atelectatic > self.atelectatic_range[1]] = 1
        atelectatic[atelectatic != atelectatic.min()] = 1
        atelectatic[atelectatic == atelectatic.min()] = 0
        return emphysema, ventilated, poorly_vent, atelectatic

    def LTRC_lungs(self):
        """Segments the lungs from single LTRC dataset.

        :return: bin_map (lungs)
        """
        # Initialize empty dictionary and parameter
        ori_ct = self.pat_obj.ct_data
        seg_data = self.pat_obj.seg_data
        bin_map = {}
        lungs_ct_data = {}

        print('\nCreating Binary Map [LTRC]- Lungs ' + '\n(Threshold value (Emphysema): ' + str(self.emphysema_va1) + ')'
              + '\n(Threshold value (Ventilated): ' + str(self.ventilated_val) + ')'
              + '\n(Threshold value (Poorly Vent): ' + str(self.poorly_vent_val) + ')'
              + '\n(Threshold value (Atelectatic): ' + str(self.atelectatic_val) + ')\n')

        # Read single series
        for series_num, ct_ in ori_ct.items():

            lungs_ct = ct_.copy()

            single_seg_data = seg_data[series_num]

            # Remove all organs except the lungs from the CT images
            lungs_map = np.zeros(self.pat_obj.data_shape[series_num])
            for _, org_data in single_seg_data.items():
                lungs_map = lungs_map + org_data
            lungs_ct[lungs_map == 0] = lungs_ct.min()
            series_lungs_ct = lungs_ct.copy()

            # Create the lungs binary map
            emphysema, ventilated, poorly_vent, atelectatic = self.lungs_binary_map(lungs_ct)

            lungs_areas = (emphysema * self.emphysema_va1) + (ventilated * self.ventilated_val) + \
                          (poorly_vent * self.poorly_vent_val) + (atelectatic * self.atelectatic_val)

            bin_map[series_num] = lungs_areas
            lungs_ct_data[series_num] = series_lungs_ct

        # Cache the data
        self.pat_obj.lungs_bin_map = bin_map
        self.pat_obj.lungs_ct_data = lungs_ct_data
        return bin_map

    def UMM_lungs(self):
        """Segments the lungs from single UMM dataset.

        :return: bin_map (lungs)
        """
        # Initialize the parameters and array
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_data

        print('\nCreating Binary Map [UMM]- Lungs ' + '\n(Threshold value (Emphysema): ' + str(self.emphysema_va1) + ')'
              + '\n(Threshold value (Ventilated): ' + str(self.ventilated_val) + ')'
              + '\n(Threshold value (Poorly Vent): ' + str(self.poorly_vent_val) + ')'
              + '\n(Threshold value (Atelectatic): ' + str(self.atelectatic_val) + ')\n')

        if not isinstance(ori_ct, dict):

            lungs_ct = ori_ct.copy()

            # Remove all organs except the lungs from the CT images
            lungs_ct[lungs_map == 0] = lungs_ct.min()
            series_lungs_ct = lungs_ct.copy()

            # Create the lungs binary map
            emphysema, ventilated, poorly_vent, atelectatic = self.lungs_binary_map(lungs_ct)

            lungs_areas = (emphysema * self.emphysema_va1) + (ventilated * self.ventilated_val) + \
                          (poorly_vent * self.poorly_vent_val) + (atelectatic * self.atelectatic_val)

            # Cache the data
            self.pat_obj.lungs_bin_map = lungs_areas
            self.pat_obj.lungs_ct_data = series_lungs_ct
            bin_map = lungs_areas
        else:
            bin_map = {}
            lungs_ct_data = {}

            for series_date, ct_ in ori_ct.items():

                lungs_ct = ct_.copy()

                # Remove all organs except the lungs from the CT images
                lungs_ct[lungs_map[series_date] == 0] = lungs_ct.min()
                series_lungs_ct = lungs_ct.copy()

                # Create the lungs binary map
                emphysema, ventilated, poorly_vent, atelectatic = self.lungs_binary_map(lungs_ct)

                lungs_areas = (emphysema * self.emphysema_va1) + (ventilated * self.ventilated_val) + \
                              (poorly_vent * self.poorly_vent_val) + (atelectatic * self.atelectatic_val)

                bin_map[series_date] = lungs_areas
                lungs_ct_data[series_date] = series_lungs_ct

            # Cache the data
            self.pat_obj.lungs_bin_map = bin_map
            self.pat_obj.lungs_ct_data = lungs_ct_data
        return bin_map

    def UKSH_lungs (self):
        """Segments the lungs from single UKSH dataset.

        :return: bin_map (lungs)
        """
        # Initialize the parameters and array
        ori_ct = self.pat_obj.ct_data
        lungs_map = self.pat_obj.lungs_data

        print('\nCreating Binary Map [UKSH]- Lungs ' + '\n(Threshold value (Emphysema): ' + str(self.emphysema_va1) + ')'
              + '\n(Threshold value (Ventilated): ' + str(self.ventilated_val) + ')'
              + '\n(Threshold value (Poorly Vent): ' + str(self.poorly_vent_val) + ')'
              + '\n(Threshold value (Atelectatic): ' + str(self.atelectatic_val) + ')\n')

        # Apply lung window for the CT image
        lungs_ct = ori_ct.copy()

        # Remove all organs except the lungs from the CT images
        lungs_ct[lungs_map == 0] = lungs_ct.min()
        series_lungs_ct = lungs_ct.copy()

        # Create the lungs binary map
        emphysema, ventilated, poorly_vent, atelectatic = self.lungs_binary_map(lungs_ct)

        lungs_areas = (emphysema * self.emphysema_va1) + (ventilated * self.ventilated_val) + \
                      (poorly_vent * self.poorly_vent_val) + (atelectatic * self.atelectatic_val)

        # Cache the data
        self.pat_obj.lungs_bin_map = lungs_areas
        self.pat_obj.lungs_ct_data = series_lungs_ct
        return lungs_areas

    def binary_map(self):
        """Performs the lungs segmentation depending on the object.

        :return: self
        """
        if self.pat_obj.data_name == 'LTRC':
            _ = self.LTRC_lungs()
        elif self.pat_obj.data_name == 'UMM':
            _ = self.UMM_lungs()
        elif self.pat_obj.data_name == 'UKSH':
            _ = self.UKSH_lungs()
        return self
