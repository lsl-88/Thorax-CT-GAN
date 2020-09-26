from Dataset import Dataset
from Processing.Lungs import Lungs
from Processing.LunglessCT import LunglessCT
from Processing.Bones import Bones
from Processing.Tissues import Tissues
from Processing.Fats import Fats

import numpy as np
from scipy.ndimage import median_filter


class PreProcess:
    """PreProcessing by creating the label map from the binary maps."""
    def __init__(self, emphysema_va1=50, ventilated_val=300, poorly_vent_val=700, atelectatic_val=1000, bones_val=1200,
                 tissue_val=1050, fats_val=650):
        """Initialize the parameters for the preprocessing of the label map.

        :param emphysema_va1: Weighted value for the lung binary map
        :param ventilated_val: Weighted value for the lung binary map
        :param poorly_vent_val: Weighted value for the lung binary map
        :param atelectatic_val: Weighted value for the lung binary map
        :param bones_val: Weighted value for the bones binary map
        :param tissue_val: Weighted value for the tissues binary map
        :param fats_val: Weighted value for the fats binary map
        """
        self.bones_val = bones_val
        self.tissue_val = tissue_val
        self.fats_val = fats_val

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

        if emphysema_va1 != self._emphysema_val or ventilated_val != self._ventilated_val or \
                poorly_vent_val != self._poorly_vent_val or atelectatic_val != self._atelectatic_val:
            self.recreate_lungs = True
        else:
            self.recreate_lungs = False

    def __repr__(self):
        return '{self.__class__.__name__}(emphysema_va1={self.emphysema_va1}, ventilated_val=' \
               '{self.ventilated_val}, poorly_vent_val={self.poorly_vent_val}, atelectatic_val={self.atelectatic_val},'\
               ' bones_val={self.bones_val}, tissue_val={self.tissue_val}, fats_val={self.fats_val})'.format(self=self)

    def LTRC_label_map(self, pat_obj):
        """Creates the label map for LTRC dataset.

        :param pat_obj: The object of LTRC class
        :return: source_data (LTRC)
        """
        # Initialize the parameters and empty dictionary
        pat_obj.source_data = None
        pat_obj.target_data = None
        lungs_map = pat_obj.lungs_bin_map
        lungless_ct = pat_obj.lungless_ct
        lungs_data = pat_obj.lungs_ct_data
        bones_data = pat_obj.bones_bin_map
        tissues_data = pat_obj.tissues_bin_map
        fats_data = pat_obj.fats_bin_map
        source_data, target_data = {}, {}

        # Read single series
        for series_num, series_lungs_map in lungs_map.items():

            # Initialize empty base map
            base_map = np.zeros(pat_obj.data_shape[series_num])

            # Combine the bones to the base map
            series_bones_data = bones_data[series_num]
            base_map = base_map + (series_bones_data * self.bones_val)

            # Combine the soft tissues to the base map
            series_tissues_data = tissues_data[series_num]
            base_map = base_map + (series_tissues_data * self.tissue_val)

            # Combine the fats to the base map
            series_fats_data = fats_data[series_num]
            base_map = base_map + (series_fats_data * self.fats_val)

            # Read the lungless CT data and lungs data
            series_ct_data = lungless_ct[series_num]
            series_lungs = lungs_data[series_num]

            # Roll the axis to (num samples, ht, wt, ch)
            base_map = np.rollaxis(base_map[np.newaxis], axis=0, start=3)
            base_map = np.rollaxis(base_map, axis=3, start=0)

            series_map = np.rollaxis(series_lungs_map[np.newaxis], axis=0, start=3)
            series_map = np.rollaxis(series_map, axis=3, start=0)

            series_ct = np.rollaxis(series_ct_data[np.newaxis], axis=0, start=3)
            series_ct = np.rollaxis(series_ct, axis=3, start=0)

            series_lungs = np.rollaxis(series_lungs[np.newaxis], axis=0, start=3)
            series_lungs = np.rollaxis(series_lungs, axis=3, start=0)

            # Create the source and target map
            source_map = np.concatenate((base_map, series_map), axis=3)
            target_map = np.concatenate((series_ct, series_lungs), axis=3)

            source_data[series_num] = source_map
            target_data[series_num] = target_map

        # Cache the data
        pat_obj.source_data = source_data
        pat_obj.target_data = target_data
        return source_data

    def UMM_label_map(self, pat_obj):
        """Creates the label map for UMM dataset.

        :param pat_obj: The object of UMM class
        :return: source_data (UMM)
        """
        # Initialize the parameters
        pat_obj.source_data = None
        pat_obj.target_data = None
        lungs_map = pat_obj.lungs_bin_map
        lungless_ct = pat_obj.lungless_ct
        lungs_data = pat_obj.lungs_ct_data
        bones_data = pat_obj.bones_bin_map
        tissues_data = pat_obj.tissues_bin_map
        fats_data = pat_obj.fats_bin_map

        if not isinstance(lungs_map, dict):

            # Initialize empty base map
            base_map = np.zeros(pat_obj.data_shape)

            # Combine the bones to the base map
            series_bones_data = bones_data
            base_map = base_map + (series_bones_data * self.bones_val)

            # Combine the soft tissues to the base map
            series_tissues_data = tissues_data
            base_map = base_map + (series_tissues_data * self.tissue_val)

            # Combine the fats to the base map
            series_fats_data = fats_data
            base_map = base_map + (series_fats_data * self.fats_val)

            # Roll the axis to (num samples, ht, wt, ch)
            base_map = np.rollaxis(base_map[np.newaxis], axis=0, start=3)
            base_map = np.rollaxis(base_map, axis=3, start=0)

            lungs = np.rollaxis(lungs_map[np.newaxis], axis=0, start=3)
            lungs = np.rollaxis(lungs, axis=3, start=0)

            ct = np.rollaxis(lungless_ct[np.newaxis], axis=0, start=3)
            ct = np.rollaxis(ct, axis=3, start=0)

            lungs_ct = np.rollaxis(lungs_data[np.newaxis], axis=0, start=3)
            lungs_ct = np.rollaxis(lungs_ct, axis=3, start=0)

            # Create the source and target map
            source_map = np.concatenate((base_map, lungs), axis=3)
            target_map = np.concatenate((ct, lungs_ct), axis=3)

            # Cache the data
            pat_obj.source_data = source_map
            pat_obj.target_data = target_map
            source_data = source_map
        else:
            source_data, target_data = {}, {}

            for series_date, series_lungs_map in lungs_map.items():

                # Initialize empty base map
                base_map = np.zeros(pat_obj.data_shape[series_date])

                # Combine the bones to the base map
                series_bones_data = bones_data[series_date]
                base_map = base_map + (series_bones_data * self.bones_val)

                # Combine the soft tissues to the base map
                series_tissues_data = tissues_data[series_date]
                base_map = base_map + (series_tissues_data * self.tissue_val)

                # Combine the fats to the base map
                series_fats_data = fats_data[series_date]
                base_map = base_map + (series_fats_data * self.fats_val)

                # Read the lungless CT data and lungs data
                series_ct_data = lungless_ct[series_date]
                series_lungs = lungs_data[series_date]

                # Roll the axis to (num samples, ht, wt, ch)
                base_map = np.rollaxis(base_map[np.newaxis], axis=0, start=3)
                base_map = np.rollaxis(base_map, axis=3, start=0)

                series_map = np.rollaxis(series_lungs_map[np.newaxis], axis=0, start=3)
                series_map = np.rollaxis(series_map, axis=3, start=0)

                series_ct = np.rollaxis(series_ct_data[np.newaxis], axis=0, start=3)
                series_ct = np.rollaxis(series_ct, axis=3, start=0)

                series_lungs = np.rollaxis(series_lungs[np.newaxis], axis=0, start=3)
                series_lungs = np.rollaxis(series_lungs, axis=3, start=0)

                # Create the source and target map
                source_map = np.concatenate((base_map, series_map), axis=3)
                target_map = np.concatenate((series_ct, series_lungs), axis=3)

                source_data[series_date] = source_map
                target_data[series_date] = target_map

            # Cache the data
            pat_obj.source_data = source_data
            pat_obj.target_data = target_data
        return source_data

    def UKSH_label_map(self, pat_obj):
        """Creates the label map for UKSH dataset.

        :param pat_obj: Object of UKSH class
        :return: source_data (UKSH)
        """
        # Initialize the parameters
        pat_obj.source_data = None
        pat_obj.target_data = None
        lungs_map = pat_obj.lungs_bin_map
        lungless_ct = pat_obj.lungless_ct
        lungs_data = pat_obj.lungs_ct_data
        bones_data = pat_obj.bones_bin_map
        tissues_data = pat_obj.tissues_bin_map
        fats_data = pat_obj.fats_bin_map

        # Initialize empty base map
        base_map = np.zeros(pat_obj.data_shape)

        # Combine the bones to the base map
        series_bones_data = bones_data
        base_map = base_map + (series_bones_data * self.bones_val)

        # Combine the soft tissues to the base map
        series_tissues_data = tissues_data
        base_map = base_map + (series_tissues_data * self.tissue_val)

        # Combine the fats to the base map
        series_fats_data = fats_data
        base_map = base_map + (series_fats_data * self.fats_val)

        # Roll the axis to (num samples, ht, wt, ch)
        base_map = np.rollaxis(base_map[np.newaxis], axis=0, start=3)
        base_map = np.rollaxis(base_map, axis=3, start=0)

        lungs = np.rollaxis(lungs_map[np.newaxis], axis=0, start=3)
        lungs = np.rollaxis(lungs, axis=3, start=0)

        ct = np.rollaxis(lungless_ct[np.newaxis], axis=0, start=3)
        ct = np.rollaxis(ct, axis=3, start=0)

        lungs_ct = np.rollaxis(lungs_data[np.newaxis], axis=0, start=3)
        lungs_ct = np.rollaxis(lungs_ct, axis=3, start=0)

        # Create the source and target map
        source_map = np.concatenate((base_map, lungs), axis=3)
        target_map = np.concatenate((ct, lungs_ct), axis=3)

        # Cache the data
        pat_obj.source_data = source_map
        pat_obj.target_data = target_map
        return source_map

    def full_label_map(self, pat_obj, bones_threshold=150, tissues_threshold=0, fats_threshold=-400):
        """Creates the label map using all the binary maps.

        :param pat_obj: Object of either LTRC, UMM or UKSH class
        :param bones_threshold: The threshold value to separate the bone from CT image (Default is 150)
        :param tissues_threshold: The threshold value to separate the soft tissue from CT image (Default is 0)
        :param fats_threshold: The threshold value to separate the fats and muscles from CT image (Default is -400)
        :return: full label map and instance to the binary map (i.e. 'self')
        """
        assert (isinstance(pat_obj, Dataset)), 'Object is not instance of Dataset'

        # Check if all binary maps are available
        if hasattr(pat_obj, 'lungs_bin_map') is False or self.recreate_lungs is True:
            pat_obj.lungs_bin_map = Lungs(pat_obj, emphysema_va1=self.emphysema_va1, ventilated_val=self.ventilated_val,
                                          poorly_vent_val=self.poorly_vent_val, atelectatic_val=self.atelectatic_val)\
                .binary_map().pat_obj.lungs_bin_map
        if hasattr(pat_obj, 'bones_bin_map') is False or bones_threshold != self._bones_thres:
            pat_obj.bones_bin_map = Bones(pat_obj, bones_threshold).binary_map().pat_obj.bones_bin_map
        if hasattr(pat_obj, 'tissues_bin_map') is False or tissues_threshold != self._tissues_thres:
            pat_obj.tissues_bin_map = Tissues(pat_obj, tissues_threshold).binary_map().pat_obj.tissues_bin_map
        if hasattr(pat_obj, 'fats_bin_map') is False or fats_threshold != self._fats_thres:
            pat_obj.fats_bin_map = Fats(pat_obj, fats_threshold).binary_map().pat_obj.fats_bin_map
        if hasattr(pat_obj, 'lungless_ct') is False:
            pat_obj.lungless_ct = LunglessCT(pat_obj).binary_map().pat_obj.lungless_ct

        if pat_obj.data_name == 'LTRC':
            print('\nCreating Label Map [LTRC]')
            _ = self.LTRC_label_map(pat_obj)
            print('\nLabel Map [LTRC] created\n')
        elif pat_obj.data_name == 'UMM':
            print('\nCreating Label Map [UMM]')
            _ = self.UMM_label_map(pat_obj)
            print('\nLabel Map [UMM] created\n')
        elif pat_obj.data_name == 'UKSH':
            print('\nCreating Label Map [UKSH]')
            _ = self.UKSH_label_map(pat_obj)
            print('\nLabel Map [UKSH] created\n')
        return self

    @property
    def print_parameters(self):
        print("\nEmphysema Values: {val}".format(val=self.emphysema_va1))
        print("Ventilated Values: {val}".format(val=self.ventilated_val))
        print("Poorly Vent Values: {val}".format(val=self.poorly_vent_val))
        print("Atelectatic Values: {val}".format(val=self.atelectatic_val))
        pass
