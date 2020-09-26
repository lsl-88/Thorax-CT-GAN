from Evaluation import Evaluation

import os
import os.path as op
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf


class MSSIM(Evaluation):
    """MSSIM evaluation for generated images."""
    def __init__(self, name, lungs_model, organs_model, merge_channel, root_dir, use_tgt_organs):
        super().__init__(name, lungs_model, organs_model, merge_channel, root_dir)

        self.use_tgt_organs = use_tgt_organs
        if self.merge_channel is True:
            self.model = lungs_model
            self.model = organs_model

        self.eval_results_dir = None
        self.create_save_directories()

    def create_save_directories(self):
        """Creates the save directories for evaluation results.

        :return: None
        """
        self.eval_results_dir = op.join(self.root_dir, 'logs', self.name, 'evaluation_results')
        if not op.exists(self.eval_results_dir):
            os.mkdir(self.eval_results_dir)
        return self

    @staticmethod
    def check_lungs(image, percentage):
        """Check each CT slice if there lungs inside the CT slice.

        :param image: Single CT slice
        :param percentage: The cutoff percentage
        :return: checker
        """
        # Total number of pixels
        total_pixels = image.shape[1] * image.shape[2]

        # Number of pixels that are lungs
        num_pixels = image[image != image.min()]

        if len(num_pixels) >= (percentage * total_pixels):
            checker = True
        else:
            checker = False
        return checker

    def results_1(self):
        """Compute the grand average MSSIM score for Framework 2 model 1 (Two channels label map) and save the results
         in the save directory.

        :return: None
        """
        # Obtain the test data list
        test_ct_list, test_lm_list = super().obtain_data_list()

        # Set the processed data directory
        source_data_dir = op.join(self.root_dir, 'Processed_data', 'source_data_2')
        target_data_dir = op.join(self.root_dir, 'Processed_data', 'target_data_2')

        # Initialize empty data frame
        col_names = ['Data', 'Average Score', 'Grand Average']
        df = pd.DataFrame(columns=col_names)
        grand_avg_score = {}
        total_dataset = len(test_lm_list)

        for single_lm, single_ct in zip(test_lm_list, test_ct_list):

            data_name = single_lm.split('_')[0]
            condition = single_lm.split('_')[1]
            data_id = single_lm.split('_')[2]
            if data_name == 'LTRC':
                series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4]
                print('\nComputing MSSIM results: {} ({}) - {} ({})'.format(data_name, condition, data_id, series_name) + '\n')
            else:
                series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4] + '_' + single_lm.split('_')[5]
                print('\nComputing MSSIM results: {} ({}) - {} (Series: {})'.format(data_name, condition, data_id, series_name) + '\n')

            full_name = data_name + '_' + condition + '_' + data_id + '_' + series_name

            # Load the source and target data
            X1 = nib.load(op.join(source_data_dir, single_lm)).get_fdata()
            X2 = nib.load(op.join(target_data_dir, single_ct)).get_fdata()

            # Generate image from source and target data
            X1_lungs = X1[:, :, :, 1]
            X1_lungs = X1_lungs[:, :, :, np.newaxis]
            X1_organs = X1[:, :, :, 0]
            X1_organs = X1_organs[:, :, :, np.newaxis]

            X2_lungs = X2[:, :, :, 1]
            X2_lungs = X2_lungs[:, :, :, np.newaxis]
            X2_organs = X2[:, :, :, 0]
            X2_organs = X2_organs[:, :, :, np.newaxis]

            # Initialize total score
            total_score = 0
            total_slice = len(X1)
            avg_score_dict = {}

            for src_organs_slice, src_lungs_slice, tgt_organs_slice, tgt_lungs_slice in zip(X1_organs, X1_lungs, X2_organs, X2_lungs):

                # Generate image for single slice
                gen_organs = self.organs_model.predict(src_organs_slice[np.newaxis])
                gen_lungs = self.lungs_model.predict(src_lungs_slice[np.newaxis])

                # Attenuate the lungs area to remove artifacts
                gen_organs[tgt_lungs_slice[np.newaxis] != tgt_lungs_slice[np.newaxis].min()] = gen_organs.min()

                # Scale to [0 , 1]
                gen_img = gen_organs + gen_lungs
                gen_img = gen_img + 1
                gen_img = gen_img / gen_img.max()

                tgt_img = tgt_organs_slice + tgt_lungs_slice
                tgt_img = tgt_img[np.newaxis, :, :, :]
                tgt_img = tgt_img + 1
                tgt_img = tgt_img / tgt_img.max()

                # Cast to tf
                gen_img = tf.cast(gen_img, dtype=tf.float32)
                tgt_img = tf.cast(tgt_img, dtype=tf.float32)

                # Compute the MSSIM score for individual slice
                score = tf.image.ssim_multiscale(gen_img, tgt_img, max_val=1.0).numpy()[0]

                # Compute the total MSSIM score for all the slices in the series
                total_score += score

            # Compute the average MSSIM score
            avg_score = total_score / total_slice
            print('Average score: ', avg_score)

            avg_score_dict['Data'] = full_name
            avg_score_dict['Average Score'] = avg_score
            df = df.append(avg_score_dict, ignore_index=True)

        # Compute the grand average MSSIM score
        grand_avg = sum(df['Average Score']) / total_dataset
        grand_avg_score['Grand Average'] = grand_avg
        df = df.append(grand_avg_score, ignore_index=True)
        df.to_csv(op.join(self.eval_results_dir, 'multiscale_ssim.csv'))
        pass

    def results_2(self):
        """Compute the grand average MSSIM score for Framework 1 (Single channel label map) and save the results in the
        save directory.

        :return: None
        """
        # Obtain the test data list
        test_ct_list, test_lm_list = super().obtain_data_list()

        # Set the processed data directory
        source_data_dir = op.join(self.root_dir, 'Processed_data', 'source_data_2')
        target_data_dir = op.join(self.root_dir, 'Processed_data', 'target_data_2')

        # Initialize empty data frame
        col_names = ['Data', 'Average Score', 'Grand Average']
        df = pd.DataFrame(columns=col_names)
        grand_avg_score = {}
        total_dataset = len(test_lm_list)

        for single_lm, single_ct in zip(test_lm_list, test_ct_list):

            data_name = single_lm.split('_')[0]
            condition = single_lm.split('_')[1]
            data_id = single_lm.split('_')[2]
            if data_name == 'LTRC':
                series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4]
                print('\nComputing MSSIM results: {} ({}) - {} ({})'.format(data_name, condition, data_id, series_name) + '\n')
            else:
                series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4] + '_' + single_lm.split('_')[5]
                print('\nComputing MSSIM results: {} ({}) - {} (Series: {})'.format(data_name, condition, data_id, series_name) + '\n')

            full_name = data_name + '_' + condition + '_' + data_id + '_' + series_name

            # Load the source and target data
            X1 = nib.load(op.join(source_data_dir, single_lm)).get_fdata()
            X2 = nib.load(op.join(target_data_dir, single_ct)).get_fdata()

            # Merge the channels and scale to [-1, 1]
            X1[:, :, :, 0] = X1[:, :, :, 0] + 1
            X1[:, :, :, 1] = X1[:, :, :, 1] + 1
            X1 = X1[:, :, :, 0] + X1[:, :, :, 1]
            X1 = X1[:, :, :, np.newaxis]
            X1 = (X1 - (X1.max() / 2)) / (X1.max() / 2)

            X2[:, :, :, 0] = X2[:, :, :, 0] + 1
            X2[:, :, :, 1] = X2[:, :, :, 1] + 1
            X2 = X2[:, :, :, 0] + X2[:, :, :, 1]
            X2 = X2[:, :, :, np.newaxis]
            X2 = (X2 - (X2.max() / 2)) / (X2.max() / 2)

            # Initialize total score
            total_score = 0
            total_slice = len(X1)
            avg_score_dict = {}

            for single_src, single_tgt in zip(X1, X2):

                # Generate image for single slice
                single_gen = self.model.predict(single_src[np.newaxis])

                # Scale to [0 , 1]
                single_gen = single_gen + 1
                single_gen = single_gen / single_gen.max()

                single_tgt = single_tgt[np.newaxis, :, :, :]
                single_tgt = single_tgt + 1
                single_tgt = single_tgt / single_tgt.max()

                # Cast to tf
                gen_img = tf.cast(single_gen, dtype=tf.float32)
                tgt_img = tf.cast(single_tgt, dtype=tf.float32)

                # Compute the MSSIM score for individual slice
                score = tf.image.ssim_multiscale(gen_img, tgt_img, max_val=1.0).numpy()[0]

                # Compute the total MSSIM score for all the slices in the series
                total_score += score

            # Compute the average MSSIM score
            avg_score = total_score / total_slice
            print('Average score: ', avg_score)

            avg_score_dict['Data'] = full_name
            avg_score_dict['Average Score'] = avg_score
            df = df.append(avg_score_dict, ignore_index=True)

        # Compute the grand average MSSIM score
        grand_avg = sum(df['Average Score']) / total_dataset
        grand_avg_score['Grand Average'] = grand_avg
        df = df.append(grand_avg_score, ignore_index=True)
        df.to_csv(op.join(self.eval_results_dir, 'multiscale_ssim.csv'))
        pass

    def results_3(self):
        """Compute the grand average MSSIM score for Framework 2 model 2 (Two channels label map) and save the results
         in the save directory.

        :return: None
        """
        # Obtain the test data list
        test_ct_list, test_lm_list = super().obtain_data_list()

        # Set the processed data directory
        source_data_dir = op.join(self.root_dir, 'Processed_data', 'source_data_2')
        target_data_dir = op.join(self.root_dir, 'Processed_data', 'target_data_2')

        # Initialize empty data frame
        col_names = ['Data', 'Average Score', 'Grand Average']
        df = pd.DataFrame(columns=col_names)
        grand_avg_score = {}
        total_dataset = len(test_lm_list)

        for single_lm, single_ct in zip(test_lm_list, test_ct_list):

            data_name = single_lm.split('_')[0]
            condition = single_lm.split('_')[1]
            data_id = single_lm.split('_')[2]
            if data_name == 'LTRC':
                series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4]
                print('\nComputing MSSIM results: {} ({}) - {} ({})'.format(data_name, condition, data_id, series_name) + '\n')
            else:
                series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4] + '_' + single_lm.split('_')[5]
                print('\nComputing MSSIM results: {} ({}) - {} (Series: {})'.format(data_name, condition, data_id, series_name) + '\n')

            full_name = data_name + '_' + condition + '_' + data_id + '_' + series_name

            # Load the source and target data
            X1 = nib.load(op.join(source_data_dir, single_lm)).get_fdata()
            X2 = nib.load(op.join(target_data_dir, single_ct)).get_fdata()

            # Generate image from source and target data
            X1_lungs = X1[:, :, :, 1]
            X1_lungs = X1_lungs[:, :, :, np.newaxis]
            X1_organs = X1[:, :, :, 0]
            X1_organs = X1_organs[:, :, :, np.newaxis]

            X2_lungs = X2[:, :, :, 1]
            X2_lungs = X2_lungs[:, :, :, np.newaxis]
            X2_organs = X2[:, :, :, 0]
            X2_organs = X2_organs[:, :, :, np.newaxis]

            # Initialize total score
            total_score = 0
            total_slice = 0
            avg_score_dict = {}

            for src_organs_slice, src_lungs_slice, tgt_organs_slice, tgt_lungs_slice in zip(X1_organs, X1_lungs, X2_organs, X2_lungs):

                # Generate image for single slice
                checker = self.check_lungs(tgt_lungs_slice, percentage=0.05)

                if checker:
                    total_slice = total_slice + 1
                    gen_lungs = self.lungs_model.predict(src_lungs_slice[np.newaxis])

                    # Scale to [0 , 1]
                    gen_img = tgt_organs_slice + gen_lungs
                    gen_img = gen_img + 1
                    gen_img = gen_img / gen_img.max()

                    tgt_img = tgt_organs_slice + tgt_lungs_slice
                    tgt_img = tgt_img[np.newaxis, :, :, :]
                    tgt_img = tgt_img + 1
                    tgt_img = tgt_img / tgt_img.max()

                    # Cast to tf
                    gen_img = tf.cast(gen_img, dtype=tf.float32)
                    tgt_img = tf.cast(tgt_img, dtype=tf.float32)

                    # Compute the MSSIM score for individual slice
                    score = tf.image.ssim_multiscale(gen_img, tgt_img, max_val=1.0).numpy()[0]

                    # Compute the total MSSIM score for all the slices in the series
                    total_score += score

            # Compute the average MSSIM score
            avg_score = total_score / total_slice
            print('Average score: ', avg_score)

            avg_score_dict['Data'] = full_name
            avg_score_dict['Average Score'] = avg_score
            df = df.append(avg_score_dict, ignore_index=True)

        # Compute the grand average MSSIM score
        grand_avg = sum(df['Average Score']) / total_dataset
        grand_avg_score['Grand Average'] = grand_avg
        df = df.append(grand_avg_score, ignore_index=True)
        df.to_csv(op.join(self.eval_results_dir, 'multiscale_ssim_use_target_organs.csv'))
        pass

    def compute_results(self):

        if self.merge_channel:
            self.results_2()
        elif not self.merge_channel and not self.use_tgt_organs:
            self.results_1()
        elif not self.merge_channel and self.use_tgt_organs:
            self.results_3()
        pass


