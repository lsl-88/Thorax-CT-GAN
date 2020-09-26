from Plot import Plot

import os
import os.path as op
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import median_filter


class Inference(Plot):
    """Plots the inference CT data and multiplane view (sagittal, axial and coronal plane)"""
    def __init__(self, name, pat_id, series, root_dir, merge_channel, model_lungs=None, model_organs=None):

        super().__init__(name, pat_id, series, root_dir)
        self.model_lungs = model_lungs
        self.model_organs = model_organs
        self.merge_channel = merge_channel
        if self.merge_channel:
            self.model = model_lungs
            self.model = model_organs

        # Variables to store data
        self.image_dir = None
        self.inference_img_dir = None
        self.inference_series_dir = None

        self.create_save_directories()

    def create_save_directories(self):

        # Set the name for the directories
        self.image_dir = op.join('../logs', self.name, 'images')
        self.inference_img_dir = op.join(self.image_dir, 'inference')
        self.inference_series_dir = op.join(self.inference_img_dir, 'series')

        if not op.exists(self.inference_series_dir):
            if not op.exists(self.inference_img_dir):
                if not op.exists(self.image_dir):
                    os.mkdir(self.image_dir)
                os.mkdir(self.inference_img_dir)
            os.mkdir(self.inference_series_dir)
        return self

    def plot_images(self, use_original_ct=False, slice_num=None, save=False):
        """This function plots the inference CT image with the label map and CT image.

        :param use_original_ct: Specify to use original CT image or processed CT image
        :param slice_num: Specify the slice number for image plot
        :param save: To save the image
        :return: None
        """
        # Load the data
        if use_original_ct:
            X1, _ = super().load_processed_data()
            X2 = super().load_original_ct()
        else:
            X1, X2 = super().load_processed_data()

        if self.merge_channel:
            X1[:, :, :, 0] = X1[:, :, :, 0] + 1
            X1[:, :, :, 1] = X1[:, :, :, 1] + 1
            X1 = X1[:, :, :, 0] + X1[:, :, :, 1]
            X1 = X1[:, :, :, np.newaxis]
            X1 = (X1 - (X1.max() / 2)) / (X1.max() / 2)

        # Select random image from X1 and X2
        if slice_num is None:
            ix = randint(0, len(X1), 1)
            src_img, tar_img = X1[ix], X2[ix]
        elif slice_num > len(X1):
            raise ValueError('The slice number indicated is out of range.')
        else:
            ix = [slice_num]
            src_img, tar_img = X1[ix], X2[ix]

        if self.merge_channel:
            gen_img = self.model.predict(src_img)
        else:
            # Generate image from source
            src_lungs = src_img[:, :, :, 1]
            src_lungs = src_lungs[:, :, :, np.newaxis]
            src_organs = src_img[:, :, :, 0]
            src_organs = src_organs[:, :, :, np.newaxis]
            tar_lungs = tar_img[:, :, :, 1]
            tar_lungs = tar_lungs[:, :, :, np.newaxis]
            tar_organs = tar_img[:, :, :, 0]
            tar_organs = tar_organs[:, :, :, np.newaxis]

            # Prediction
            gen_lungs_img = self.model_lungs.predict(src_lungs)
            gen_organs_img = self.model_organs.predict(src_organs)

            # Attenuate the lungs area to remove artifacts
            gen_organs_img[tar_lungs != tar_lungs.min()] = gen_organs_img.min()

            # Combine the lungs and organs together
            gen_img = gen_organs_img[:, :, :, 0] + gen_lungs_img[:, :, :, 0]
            gen_img = gen_img[:, :, :, np.newaxis]

            src_img = src_img[:, :, :, 0] + src_img[:, :, :, 1]
            src_img = src_img[:, :, :, np.newaxis]

        tar_img = tar_img[:, :, :, 0] + tar_img[:, :, :, 1]
        tar_img = tar_img[:, :, :, np.newaxis]

        # Stack the source, generated and targeted into an array plot
        images = np.vstack((src_img, gen_img, tar_img))

        # Set the title
        titles = ['Source', 'Generated', 'Target']

        # Set the figure size
        plt.figure(figsize=(30, 10))

        # Plot images row by row
        for i in range(len(images)):
            # Define subplot
            plt.subplot(1, 3, 1 + i)
            plt.axis('off')

            # Plot raw pixel data
            plt.imshow(np.squeeze(images[i], axis=2), cmap='bone')

            # Show title
            plt.title(titles[i], fontsize=28)

        if save:
            if self.dataset == 'LTRC':
                file_name = self.dataset + '_' + self.pat_id + '_' + self.series + '_Src_Gen_Exp_imgs_slice_' + str(
                    ix) + '.png'
            else:
                file_name = self.dataset + '_' + self.pat_id + '_Series_' + self.series + '_Src_Gen_Exp_imgs_slice_' + str(
                    ix) + '.png'
            plt.savefig(op.join(self.inference_img_dir, file_name))
        plt.show()
        pass

    def series_prediction(self, save=False):
        """This function predicts for the full series.

        :param save: To save the image
        :return: None
        """
        # Load the data
        X1, X2 = super().load_processed_data()

        if self.merge_channel:
            X1[:, :, :, 0] = X1[:, :, :, 0] + 1
            X1[:, :, :, 1] = X1[:, :, :, 1] + 1
            X1 = X1[:, :, :, np.newaxis]
            X1 = (X1 - (X1.max() / 2)) / (X1.max() / 2)

        # Initialize empty array
        inference_array = np.array([]).reshape(0, X1.shape[1], X1.shape[2], 1)

        if self.dataset == 'LTRC':
            print('Performing inference on: ' + self.dataset + ' - ' + self.pat_id + ' (' + self.series + ')')
        else:
            print('Performing inference on: ' + self.dataset + ' - ' + self.pat_id + ' (Series - ' + self.series + ')')

        if self.merge_channel:
            for single_src_slice in X1:
                gen_img = self.model.predict(single_src_slice[np.newaxis])
                inference_array = np.concatenate((inference_array, gen_img), axis=0)
        else:
            for single_src_slice, single_tgt_slice in zip(X1, X2):
                # Generate image for single slice
                single_src_slice_lungs = single_src_slice[:, :, 1]
                single_src_slice_organs = single_src_slice[:, :, 0]
                single_tgt_slice_lungs = single_tgt_slice[:, :, 1]
                gen_lungs_img = self.model_lungs.predict(single_src_slice_lungs[np.newaxis, :, :, np.newaxis])
                gen_organs_img = self.model_organs.predict(single_src_slice_organs[np.newaxis, :, :, np.newaxis])

                # Attenuate the lungs area to remove artifacts
                gen_organs_img[single_tgt_slice_lungs != single_tgt_slice_lungs.min()] = gen_organs_img.min()

                # Combine the lungs and organs to form single image
                gen_img = gen_organs_img[:, :, :, 0] + gen_lungs_img[:, :, :, 0]
                gen_img = gen_img[:, :, :, np.newaxis]

                inference_array = np.concatenate((inference_array, gen_img), axis=0)

        if save:
            series_inf_ct = nib.Nifti1Pair(inference_array, self.affine)
            if self.dataset == 'LTRC':
                file_name = self.dataset + '_' + self.pat_id + '_' + self.series + '.nii'
            else:
                file_name = self.dataset + '_' + self.pat_id + '_Series_' + self.series + '.nii'

            nib.save(series_inf_ct, op.join(self.inference_series_dir, file_name))
        pass

    def load_inference(self):
        """This function loads the inference data.

        :return: img (Inference CT)
        """
        _, _ = super().load_processed_data()

        if self.dataset == 'LTRC':
            file_name = self.dataset + '_' + self.pat_id + '_' + self.series + '.nii'
        else:
            file_name = self.dataset + '_' + self.pat_id + '_Series_' + self.series + '.nii'

        # Load the inference data
        img = nib.load(op.join(self.inference_series_dir, file_name)).get_fdata()
        img = np.squeeze(img, axis=3)
        img = np.rollaxis(img, axis=0, start=3)
        return img

    def multiplane_views(self, sagittal_slice=None, axial_slice=None, coronal_slice=None, save=False):
        """This function plots the multiplane views of the single inference CT image.

        :param sagittal_slice: Specify the sagittal slice number for image plot
        :param axial_slice: Specify the axial slice number for image plot
        :param coronal_slice: Specify the coronal slice number for image plot
        :param save: To save the image
        :return: None
        """
        # Load the data
        image = self.load_inference()

        # Select random image the inference data
        if sagittal_slice is None:
            sagittal_slice = randint(0, image.shape[0])
        elif sagittal_slice > image.shape[0]:
            raise ValueError('The sagittal slice number indicated is out of range.')

        if axial_slice is None:
            axial_slice = randint(0, image.shape[2])
        elif axial_slice > image.shape[2]:
            raise ValueError('The axial slice number indicated is out of range.')

        if coronal_slice is None:
            coronal_slice = randint(0, image.shape[1])
        elif coronal_slice > image.shape[1]:
            raise ValueError('The coronal slice number indicated is out of range.')

        # Set the multiplane view images
        sagittal_image = image[sagittal_slice, :, :]  # Axis 0
        axial_image = image[:, :, axial_slice]  # Axis 2
        coronal_image = image[:, coronal_slice, :]  # Axis 1

        sagittal_image = resize(sagittal_image, output_shape=(512,512), anti_aliasing=True)
        axial_image = resize(axial_image, output_shape=(512,512), anti_aliasing=True)
        coronal_image = resize(coronal_image, output_shape=(512,512), anti_aliasing=True)

        plt.figure(figsize=(30, 10))
        plt.suptitle(self.dataset + ' - ' + self.pat_id, fontsize=24, y=0.95)

        plt.subplot(1,3,1)
        plt.imshow(np.rot90(sagittal_image), cmap='bone')
        plt.title('Sagittal Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(np.rot90(axial_image), cmap='bone')
        plt.title('Axial Plane', fontsize=18)
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(np.rot90(coronal_image), cmap='bone')
        plt.title('Coronal Plane', fontsize=18)
        plt.axis('off')

        if save:
            file_name = self.dataset+'_'+self.pat_id+'_'+self.series+'_slice_['+str(sagittal_slice)+'-'+str(axial_slice)\
                        +'-'+str(coronal_slice)+']_multiplane_view.png'
            plt.savefig(op.join(self.inference_img_dir, file_name))
        plt.show()
        pass
