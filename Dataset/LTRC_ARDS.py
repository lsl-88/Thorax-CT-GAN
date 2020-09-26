from Dataset import LTRC


class LTRC_ARDS(LTRC):
    """Single LTRC (ARDS) dataset consists of CT images and its corresponding binary maps."""
    condition = 'ARDS'
    _cache_list = []

    def __init__(self, data_id, data_dir, **kwargs):
        """Initialize a LTRC (ARDS) dataset without loading it.

        :param data_id: String identifier for the dataset, e.g. '102022'
        :param data_dir: The path to the data directory in which the LTRC dataset resides
        :param **kwargs: Arbitrary keyword arguments (unused)
        """
        super().__init__(data_id, data_dir, **kwargs)

        self.cache(data_id)

    def __repr__(self):
        return '{self.__class__.__name__}(data_id={self.data_id}, data_dir={self.data_dir}, **kwargs)' \
            .format(self=self)

    def cache(self, data_id):
        if data_id not in self._cache_list:
            self._cache_list.append(data_id)
        return self

    def instances(self):
        return len(self._cache_list)

    def load(self, data_type='all', verbose=True, **kwargs):
        """Loads a series dataset.

        :param data_type: Data type to load ('CT', 'seg' or 'all')
        :param verbose: Print out the progress of loading
        :param kwargs: Arbitrary keyword arguments (unused)
        :return: Instance to the dataset (i.e. 'self')
        """
        assert data_type == 'CT' or data_type == 'seg' or data_type == 'all', 'Please select the appropriate data type'

        # Load CT data only
        if data_type == 'CT':
            if verbose:
                print('\nLoading: ' + str(self.data_name) + '-' + str(self.data_id) + ' - CT data')
            super().load_ct(self.data_id)

        # Load Seg data only
        elif data_type == 'seg':
            if verbose:
                print('\nLoading: ' + str(self.data_name) + '-' + str(self.data_id) + ' - segmentation data')
            super().load_seg(self.data_id)

        # Load both CT and Seg data
        elif data_type == 'all':
            if verbose:
                print('\nLoading: ' + str(self.data_name) + '-' + str(self.data_id))
            super().load_ct(self.data_id)
            super().load_seg(self.data_id)
        return self
