import os
import os.path as op
from glob import glob
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize


class Plot:
    """Plots the original CT data, processed data and multiplane view (sagittal, axial and coronal plane)"""
    def __init__(self, name, pat_id, series, root_dir):
        """Initialize the details of the dataset for the plot.

        :param name: Name of the run, e.g. 'Run_1'
        :param pat_id: String identifier for the dataset, e.g. 'pat0001'
        :param series: Name of the series
        :param root_dir: Path to the root directory
        """
        self.name = name
        self.pat_id = pat_id
        self.series = series
        self.root_dir = root_dir

        # Variables to store data
        self.image_dir = None
        self.original_img_dir = None
        self.processed_img_dir = None
        self.multiplane_img_dir = None
        self.dataset = None
        self.affine = None

        self.create_save_directories()

    def create_save_directories(self):

        # Set the name for the directories
        self.image_dir = op.join('../logs', self.name, 'images')
        self.original_img_dir = op.join(self.image_dir, 'original')
        self.processed_img_dir = op.join(self.image_dir, 'processed')
        self.multiplane_img_dir = op.join(self.image_dir, 'multiplane')

        if not op.exists(self.original_img_dir):
            if not op.exists(self.image_dir):
                os.mkdir(self.image_dir)
            os.mkdir(self.original_img_dir)

        if not op.exists(self.processed_img_dir):
            if not op.exists(self.image_dir):
                os.mkdir(self.image_dir)
            os.mkdir(self.processed_img_dir)

        if not op.exists(self.multiplane_img_dir):
            if not op.exists(self.image_dir):
                os.mkdir(self.image_dir)
            os.mkdir(self.multiplane_img_dir)
        return self

    def load_processed_data(self):
        """This function loads the processed data.

        :return: X1, X2 (Label map and CT respectively)
        """
        # Set the patient data directory
        processed_data_dir = op.join(self.root_dir, 'Processed_data')
        pat_ct_dir = op.join(processed_data_dir, 'target_data_2')
        pat_label_dir = op.join(processed_data_dir, 'source_data_2')

        ct_series_file = glob(op.join(pat_ct_dir, '[LTRC-UKSH-UMM]*' + '_' + str(self.pat_id) + '*'))
        label_series_file = glob(op.join(pat_label_dir, '[LTRC-UKSH-UMM]*' + '_' + str(self.pat_id) + '*'))

        # For dataset with more than two series
        if len(ct_series_file) > 1:

            # Initialize empty list
            series_list = []
            for single_series in ct_series_file:

                dataset = single_series.split('\\')[-1].split('_')[0]
                self.dataset = dataset

                if dataset == 'LTRC':
                    series_name = single_series.split('\\')[-1].split('_')[3] + '_' + \
                                  single_series.split('\\')[-1].split('_')[4]
                else:
                    series_name = single_series.split('\\')[-1].split('_')[3] + '_' + \
                                  single_series.split('\\')[-1].split('_')[4] + '_' + \
                                  single_series.split('\\')[-1].split('_')[5]
                series_list.append(series_name)
            if self.series is None:
                raise NameError('Please specify the series - {}'.format(series_list))
            else:
                for single_ct, single_label in zip(ct_series_file, label_series_file):

                    dataset = single_ct.split('\\')[-1].split('_')[0]
                    self.dataset = dataset
                    if dataset == 'LTRC':
                        series_name = single_ct.split('\\')[-1].split('_')[3] + '_' + \
                                      single_ct.split('\\')[-1].split('_')[4]
                        if series_name == self.series:
                            ct_series_file = single_ct
                            label_series_file = single_label
                        elif self.series not in series_list:
                            raise NameError('Series - {} is not available.'.format(self.series))
                    else:
                        series_name = single_ct.split('\\')[-1].split('_')[3] + '_' + \
                                      single_ct.split('\\')[-1].split('_')[4] + '_' + \
                                      single_ct.split('\\')[-1].split('_')[5]
                        if series_name == self.series:
                            ct_series_file = single_ct
                            label_series_file = single_label
                        elif self.series not in series_list:
                            raise NameError('Series - {} is not available.'.format(self.series))
        # For dataset with only one series
        elif len(ct_series_file) == 1:
            ct_series_file = ct_series_file[0]
            label_series_file = label_series_file[0]

            self.dataset = ct_series_file.split('\\')[-1].split('_')[0]
            if self.dataset == 'LTRC':
                self.series = ct_series_file.split('\\')[-1].split('_')[3] + '_' + \
                              ct_series_file.split('\\')[-1].split('_')[4]
            else:
                self.series = ct_series_file.split('\\')[-1].split('_')[3] + '_' + \
                              ct_series_file.split('\\')[-1].split('_')[4] + '_' + \
                              ct_series_file.split('\\')[-1].split('_')[5]
        # For dataset with no series found
        else:
            raise NameError('Dataset does not exists.')

        img_label = nib.load(label_series_file)
        X1 = img_label.get_fdata()
        img_ct = nib.load(ct_series_file)
        X2 = img_ct.get_fdata()
        self.affine = img_ct.affine
        return X1, X2

    def load_original_ct(self):
        """This function loads the original CT data.

        :return: X2 (Original CT respectively)
        """
        # Set the patient data directory
        LTRC_data_dir = op.join(self.root_dir, 'Data', '[LTRC]*', self.pat_id, '*')
        UMM_data_dir = op.join(self.root_dir, 'Data', '[UMM]*', self.pat_id, 'thx_endex*')
        UKSH_data_dir = op.join(self.root_dir, 'Data', '[UKSH]*', self.pat_id, 'nifti', '*')

        LTRC_data_files = glob(LTRC_data_dir)
        UMM_data_files = glob(UMM_data_dir)
        UKSH_data_files = glob(UKSH_data_dir)

        if len(LTRC_data_files) != 0:
            self.dataset = 'LTRC'
            if len(LTRC_data_files) == 1:
                LTRC_data_files = LTRC_data_files[0]
                self.series = LTRC_data_files.split('\\')[-1].split('_')[2] + '_' + \
                              LTRC_data_files.split('\\')[-1].split('_')[3]
                ct_series_file = glob(op.join(LTRC_data_files, 'nifti', '*'))[0]
            else:
                series_list = []
                for single_file in LTRC_data_files:
                    series_name = single_file.split('\\')[-1].split('_')[2] + '_' + \
                                  single_file.split('\\')[-1].split('_')[3]
                    series_list.append(series_name)
                    if series_name == self.series:
                        single_ct_series = single_file
                        ct_series_file = glob(op.join(single_ct_series, 'nifti', '*'))[0]
                if self.series is None:
                    raise NameError('Please specify the series - {}'.format(series_list))
                elif self.series not in series_list:
                    raise NameError('Series - {} is not available.'.format(self.series))

        elif len(UMM_data_files) != 0:
            self.dataset = 'UMM'
            if len(UMM_data_files) == 1:
                UMM_data_files = UMM_data_files[0]
                self.series = '2000_01_01'
                ct_series_file = glob(op.join(UMM_data_files, 'nifti', '*'))[0]
            else:
                series_list = []
                for single_file in UMM_data_files:
                    series_name = single_file.split('/')[-1].split('_')[2] + '_' + \
                                  single_file.split('/')[-1].split('_')[3] + '_' + \
                                  single_file.split('/')[-1].split('_')[4]
                    series_list.append(series_name)
                    if series_name == self.series:
                        single_ct_series = single_file
                        ct_series_file = glob(op.join(single_ct_series, 'nifti', '*'))[0]
                if self.series is None:
                    raise NameError('Please specify the series - {}'.format(series_list))
                elif self.series not in series_list:
                    raise NameError('Series - {} is not available.'.format(self.series))

        elif len(UKSH_data_files) != 0:
            self.dataset = 'UKSH'
            if len(UKSH_data_files) == 1:
                UKSH_data_files = UKSH_data_files[0]
                self.series = '2000_01_01'
                ct_series_file = glob(op.join(UKSH_data_files))[0]

        X2 = nib.load(ct_series_file).get_fdata()
        X2 = np.rollaxis(X2, axis=2)
        X2 = np.rollaxis(X2[np.newaxis], axis=0, start=4)
        return X2

    def original_data(self, slice_num=None, save=False):
        """This function plots the single original CT data.

        :param slice_num: Specify the slice number for image plot
        :param save: To save the image
        :return: None
        """
        # Load the data
        X2 = self.load_original_ct()

        # Select the slice
        if slice_num is None:
            slice_num = randint(0, len(X2))
        elif slice_num > len(X2):
            raise ValueError('The slice number indicated is out of range.')

        # Plot the single CT image
        plt.figure(figsize=(10, 10))
        plt.imshow(np.squeeze(X2[slice_num], axis=2), cmap='bone')
        plt.axis('off')
        plt.title('{} - {} (Slice-{})'.format(self.dataset, self.pat_id, slice_num), fontsize=28)
        if save:
            print(self.dataset)
            print(self.pat_id)
            print(self.series)
            file_name = self.dataset + '_' + self.pat_id + '_' + self.series + '_slice_[' + str(slice_num) + '].png'
            plt.savefig(op.join(self.original_img_dir, file_name))
        plt.show()
        pass

    def processed_data(self, slice_num=None, save=False):
        """This function plots the label map and processed CT image.

        :param slice_num: Specify the slice number for image plot
        :param save: To save the image
        :return: None
        """
        # Load the data
        X1, X2 = self.load_processed_data()

        # Select the slice
        if slice_num is None:
            slice_num = randint(0, len(X1))
        elif slice_num > len(X1):
            raise ValueError('The slice number indicated is out of range.')

        # Plot the CT and label map images
        plt.figure(figsize=(20, 10))
        plt.suptitle('{} - {} (Slice-{})'.format(self.dataset, self.pat_id, slice_num), fontsize=32)

        plt.subplot(2, 2, 1)
        plt.imshow(X2[slice_num, :, :, 0], cmap='bone')
        plt.axis('off')
        plt.title('Target Data (First Channel)', fontsize=24)

        plt.subplot(2, 2, 2)
        plt.imshow(X1[slice_num, :, :, 0], cmap='bone')
        plt.axis('off')
        plt.title('Source Data (First Channel)', fontsize=24)

        plt.subplot(2, 2, 3)
        plt.imshow(X2[slice_num, :, :, 1], cmap='bone')
        plt.axis('off')
        plt.title('Target Data (Second Channel)', fontsize=24)

        plt.subplot(2, 2, 4)
        plt.imshow(X1[slice_num, :, :, 1], cmap='bone')
        plt.axis('off')
        plt.title('Source Data (Second Channel)', fontsize=24)
        if save:
            file_name = self.dataset + '_' + self.pat_id + '_' + self.series + '_slice_' + str(slice_num) + '.png'
            plt.savefig(op.join(self.processed_img_dir, file_name))
        plt.show()
        pass

    def multiplane_views(self, plot_target=True, sagittal_slice=None, axial_slice=None, coronal_slice=None, save=False):
        """This function plots the multiplane views of the single original or processed CT image or label map.

        :param plot_target: Specify to plot CT image or label map
        :param sagittal_slice: Specify the sagittal slice number for image plot
        :param axial_slice: Specify the axial slice number for image plot
        :param coronal_slice: Specify the coronal slice number for image plot
        :param save: To save the image
        :return: None
        """
        # Load the data
        X1, X2 = self.load_processed_data()

        # Plot the CT or label map
        if plot_target:
            image_first_ch = X2[:, :, :, 0]
            image_second_ch = X2[:, :, :, 1]
        else:
            image_first_ch = X1[:, :, :, 0]
            image_second_ch = X1[:, :, :, 1]

        # Rearrange the axis of the image array
        image_A = np.rollaxis(image_first_ch, axis=0, start=3)
        image_B = np.rollaxis(image_second_ch, axis=0, start=3)

        # Select random image the inference data
        if sagittal_slice is None:
            sagittal_slice = randint(0, image_A.shape[0])
        elif sagittal_slice > image_A.shape[0]:
            raise ValueError('The sagittal slice number indicated is out of range.')

        if axial_slice is None:
            axial_slice = randint(0, image_A.shape[1])
        elif axial_slice > image_A.shape[1]:
            raise ValueError('The axial slice number indicated is out of range.')

        if coronal_slice is None:
            coronal_slice = randint(0, image_A.shape[2])
        elif coronal_slice > image_A.shape[2]:
            raise ValueError('The coronal slice number indicated is out of range.')

        # Set the multiplane view images
        sagittal_image_A = image_A[sagittal_slice, :, :]  # Axis 0
        axial_image_A = image_A[:, :, axial_slice]  # Axis 2
        coronal_image_A = image_A[:, coronal_slice, :]  # Axis 1

        sagittal_image_B = image_B[sagittal_slice, :, :]  # Axis 0
        axial_image_B = image_B[:, :, axial_slice]  # Axis 2
        coronal_image_B = image_B[:, coronal_slice, :]  # Axis 1

        sagittal_image_A = resize(sagittal_image_A, output_shape=(512,512), anti_aliasing=True)
        axial_image_A = resize(axial_image_A, output_shape=(512,512), anti_aliasing=True)
        coronal_image_A = resize(coronal_image_A, output_shape=(512,512), anti_aliasing=True)

        sagittal_image_B = resize(sagittal_image_B, output_shape=(512,512), anti_aliasing=True)
        axial_image_B = resize(axial_image_B, output_shape=(512,512), anti_aliasing=True)
        coronal_image_B = resize(coronal_image_B, output_shape=(512,512), anti_aliasing=True)

        plt.figure(figsize=(30, 20))
        plt.suptitle(self.dataset + ' - ' + self.pat_id, fontsize=24, y=0.95)

        plt.subplot(2,3,1)
        plt.imshow(np.rot90(sagittal_image_A), cmap='bone')
        plt.title('Sagittal Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(2,3,2)
        plt.imshow(np.rot90(axial_image_A), cmap='bone')
        plt.title('Axial Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(2,3,3)
        plt.imshow(np.rot90(coronal_image_A), cmap='bone')
        plt.title('Coronal Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(2,3,4)
        plt.imshow(np.rot90(sagittal_image_B), cmap='bone')
        plt.title('Sagittal Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(2,3,5)
        plt.imshow(np.rot90(axial_image_B), cmap='bone')
        plt.title('Axial Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(2,3,6)
        plt.imshow(np.rot90(coronal_image_B), cmap='bone')
        plt.title('Coronal Plane', fontsize=18)
        plt.axis('off')

        if save:
            if plot_target:
                file_name = self.dataset+'_'+self.pat_id+'_'+self.series+'_slice_['+str(sagittal_slice)+'-'+str(axial_slice)\
                            +'-'+str(coronal_slice)+']_ct_multiplane_view.png'
            else:
                file_name = self.dataset+'_'+self.pat_id+'_'+self.series+'_slice_['+str(sagittal_slice)+'-'+str(axial_slice)\
                            +'-'+str(coronal_slice)+']_label_map_multiplane_view.png'
            plt.savefig(op.join(self.multiplane_img_dir, file_name))
        plt.show()
        pass

