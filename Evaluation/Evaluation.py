from glob import glob

import os.path as op
import pandas as pd
import nibabel as nib
import numpy as np
from sklearn.utils import shuffle


class Evaluation:
    """Base class of evaluation module."""
    def __init__(self, name, lungs_model, organs_model, merge_channel, root_dir):

        if root_dir is None:
            self.root_dir = '/home/ubuntu/sl_root'
        else:
            self.root_dir = root_dir
        self.name = name
        self.merge_channel = merge_channel
        self.lungs_model = lungs_model
        self.organs_model = organs_model
        if self.merge_channel is True:
            self.model = lungs_model
            self.model = organs_model

    def obtain_data_list(self):
        """Obtains the test data list from the Train_Test_Summary.csv.

        :return: test_ct_list, test_lm_list
        """
        # Set the directory and file name
        data_summary_dir = op.join(self.root_dir, 'logs', self.name, 'data_summary')
        file_name = 'Train_Test_Summary_generative.csv'

        # Read the csv and obtain the train data list
        df = pd.read_csv(op.join(data_summary_dir, file_name))
        train_data = df['Train Data'].values.tolist()

        # Obtain the CT and label map list and file names
        processed_data_dir = op.join(self.root_dir, 'Processed_data')
        ct_list = glob(op.join(processed_data_dir, 'target_data_2', '*'))
        lm_list = glob(op.join(processed_data_dir, 'source_data_2', '*'))

        ct_files = [single_file.split('/')[-1] for single_file in ct_list]
        lm_files = [single_file.split('/')[-1] for single_file in lm_list]
        ct_files.sort(), lm_files.sort()
        lm_files, ct_files = shuffle(lm_files, ct_files)

        # Initialize empty lists
        test_ct_list, test_lm_list = [], []

        for single_ct, single_lm in zip(ct_files, lm_files):

            condition = single_ct.split('_')[1]

            if condition == 'ARDS':
                # Append to test ARDS data to the empty list
                if single_ct not in train_data and single_lm not in train_data:
                    test_ct_list.append(single_ct)
                    test_lm_list.append(single_lm)
        return test_ct_list, test_lm_list

    def generate_synthetic_data(self, save=True):
        """Predicts the full series.

        :param save: To save the image
        :return: None
        """
        print('Performing Inference on all data.\n')

        # Obtain the test data list
        test_ct_list, test_lm_list = self.obtain_data_list()
        lm_data_dir = op.join(self.root_dir, 'Processed_data', 'source_data_2')
        ct_data_dir = op.join(self.root_dir, 'Processed_data', 'target_data_2')

        for single_lm, single_ct in zip(test_lm_list, test_ct_list):

            data_name = single_lm.split('_')[0]
            condition = single_lm.split('_')[1]
            data_id = single_lm.split('_')[2]

            if data_name == 'LTRC':
                series = single_lm.split('_')[3] + '_' + single_lm.split('_')[4]
                print('Performing inference on: ' + data_name + ' - ' + data_id + ' (' + series + ')')
            else:
                series = single_lm.split('_')[3] + '_' + single_lm.split('_')[4] + '_' + single_lm.split('_')[5]
                print('Performing inference on: ' + data_name + ' - ' + data_id + ' (Series: ' + series + ')')

            # Load the data
            img_lm = nib.load(op.join(lm_data_dir, single_lm))
            img_ct = nib.load(op.join(ct_data_dir, single_ct))
            X1 = img_lm.get_fdata()
            X2 = img_ct.get_fdata()
            affine = img_lm.affine

            if self.merge_channel is True:

                # Merge the channels and scale to [-1, 1]
                X1[:, :, :, 0] = X1[:, :, :, 0] + 1
                X1[:, :, :, 1] = X1[:, :, :, 1] + 1
                X1 = X1[:, :, :, 0] + X1[:, :, :, 1]
                X1 = X1[:, :, :, np.newaxis]
                X1 = (X1 - (X1.max() / 2)) / (X1.max() / 2)

                # Initialize empty array
                inference_array = np.array([]).reshape(0, X1.shape[1], X1.shape[2], 1)

                for single_slice in X1:
                    gen_img = self.model.predict(single_slice[np.newaxis])

                    inference_array = np.concatenate((inference_array, gen_img), axis=0)

            elif self.merge_channel is False:

                # Initialize empty array
                inference_array = np.array([]).reshape(0, X1.shape[1], X1.shape[2], 1)

                X1_organs = X1[:, :, :, 0]
                X1_organs = X1_organs[:, :, :, np.newaxis]
                X1_lungs = X1[:, :, :, 1]
                X1_lungs = X1_lungs[:, :, :, np.newaxis]
                X2_lungs = X2[:, :, :, 1]
                X2_lungs = X2_lungs[:, :, :, np.newaxis]

                for organs_slice, lungs_slice, tgt_lungs in zip(X1_organs, X1_lungs, X2_lungs):

                    # Generate image for single slice
                    gen_organs = self.organs_model.predict(organs_slice[np.newaxis])
                    gen_lungs = self.lungs_model.predict(lungs_slice[np.newaxis])

                    # Attenuate the lungs area to remove artifacts
                    tgt_lungs = tgt_lungs[np.newaxis]
                    gen_organs[tgt_lungs != tgt_lungs.min()] = gen_organs.min()

                    gen_img = gen_organs + gen_lungs

                    inference_array = np.concatenate((inference_array, gen_img), axis=0)
            else:
                raise NameError('Please specify the merge channel.')

            if save:
                series_inf_ct = nib.Nifti1Pair(inference_array, affine)
                file_name = data_name + '_' + condition + '_' + data_id + '_' + series + '_inference.nii'
                nib.save(series_inf_ct, op.join(self.inference_data_dir, file_name))
        pass










