from abc import ABC, abstractmethod


class DatasetError(Exception):
    pass


class Dataset(ABC):
    """
    Abstract base class representing a dataset. All datasets should subclass from this baseclass and need to implement
    the 'load' function. Initializing of the dataset and actually loading the data is separated as the latter may
    require significant time, depending on where the data is coming from. It also allows to implement different
    handlers for the remote end where the data originates, e.g. download from cloud server, etc.
    When subclassing from Dataset, it is helpful to set fields 'data_type', 'data_name', and 'data_id'.
    """
    def __init__(self, **kwargs):
        """Initialize a dataset."""

    @abstractmethod
    def load(self, **kwargs):
        """Load the data and prepare it for usage.

        :param kwargs: Arbitrary keyword arguments
        :return: self
        """
        return self

    @property
    def print_stats(self):
        """This function prints information about the dataset. This method uses the fields that need to be implemented
        when subclassing.
        """
        if self.data_name == 'LTRC':
            print('\nData identification: {name}-{id}'.format(name=self.data_name, id=self.data_id))

            assert self.ct_data is not None or self.seg_data is not None, 'Data has been initialized but not loaded.'

            print('Series: {series}'.format(series=self.series))
            for single_series in self.series:
                print('{type}-data [{series}] shape: {shape}'.format(type=self.data_type, series=single_series,
                                                                     shape=self.ct_data[single_series].shape))
                for single_scan_type in self.scan_type:
                    if single_scan_type != 'CT':
                        print('{type}-binary map [{series}] shape: {shape}'.format(type=single_scan_type.upper(),
                                                                                   series=single_series,
                                                                                   shape=self.seg_data[single_series][
                                                                                       single_scan_type].shape))

        elif self.data_name == 'UMM':
            print('\nData identification: {name}-{id}'.format(name=self.data_name, id=self.data_id))

            assert self.ct_data is not None or self.lungs_areas_data is not None, 'Data has been initialized but not loaded.'

            print('Series: {series}'.format(series=self.series))
            if not isinstance(self.ct_data, dict):
                print('{type}-data shape: {shape}'.format(type=self.data_type, shape=self.data_shape))
            else:
                for single_series in self.series:
                    print('{type}-data [{series}] shape: {shape}'.format(type=self.data_type, series=single_series,
                                                                         shape=self.ct_data[single_series].shape))

        elif self.data_name == 'UKSH':
            print('\nData identification: {name}-{id}'.format(name=self.data_name, id=self.data_id))

            assert self.ct_data is not None and self.lungs_areas_data is not None, 'Data has been initialized but not loaded.'

            print('Series: {series}'.format(series=self.series))
            print('{type}-data shape: {shape}'.format(type=self.data_type, shape=self.data_shape))
