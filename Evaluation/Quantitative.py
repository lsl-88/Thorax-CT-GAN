from Evaluation import Evaluation

from glob import glob
import os
import os.path as op
import nibabel as nib
import numpy as np
from skimage.transform import resize


class Quantitative(Evaluation):
    """Generation of the evaluation data for quantitative evaluation."""
    def __init__(self, name, lungs_model, organs_model, merge_channel, root_dir, eval_cls, eval_type):

        super().__init__(name, lungs_model, organs_model, merge_channel, root_dir)
        self.eval_cls = eval_cls
        self.eval_type = eval_type

        self.eval_data_dir = None
        self.inference_data_dir = None
        self.create_save_directories()

    def create_save_directories(self):
        """Creates the save directories for evaluation results.

        :return: None
        """
        # Set the name for the processed data directory and inference data directory
        self.eval_data_dir = op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls)
        self.inference_data_dir = op.join(self.root_dir, 'logs', self.name, 'Inference_data')

        if not op.exists(self.eval_data_dir):
            if not op.exists(op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type)):
                if not op.exists(op.join(self.root_dir, 'Evaluation_data', self.name)):
                    if not op.exists(op.join(self.root_dir, 'Evaluation_data')):
                        os.mkdir(op.join(self.root_dir, 'Evaluation_data'))
                    os.mkdir(op.join(self.root_dir, 'Evaluation_data', self.name))
                os.mkdir(op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type))
            os.mkdir(self.eval_data_dir)
            os.mkdir(op.join(self.eval_data_dir, 'ct_data'))
            os.mkdir(op.join(self.eval_data_dir, 'label_data'))

        if not op.exists(self.inference_data_dir):
            os.mkdir(self.inference_data_dir)
        return self

    def load_dataset(self, split_ratio=0.6):
        """Load the datasets according to the evaluation type.

        :param split_ratio: The ratio to split the train, validation and test dataset
        :return: train_dataset, val_dataset, test_dataset
        """
        # Generate the data set
        test_ct_list, test_lm_list = super().obtain_data_list()

        if self.eval_type == 'first':

            # Split the data for real CT
            train_ratio = round(len(test_ct_list) * split_ratio)
            train_ct, train_lm = test_ct_list[:train_ratio], test_lm_list[:train_ratio]

            test_val_ct, test_val_lm = test_ct_list[train_ratio:], test_lm_list[train_ratio:]
            val_ratio = len(test_val_ct) // 2

            val_ct, val_lm = test_val_ct[:val_ratio], test_val_lm[:val_ratio]
            test_ct, test_lm = test_val_ct[val_ratio:], test_val_lm[val_ratio:]

            train_dataset = train_lm, train_ct
            val_dataset = val_lm, val_ct
            test_dataset = test_lm, test_ct

        elif self.eval_type == 'second':

            # Generate the synthetic data
            if len(glob(op.join(self.inference_data_dir, '*'))) == 0:
                super().generate_synthetic_data(save=True)
            gen_data_list = glob(op.join(self.inference_data_dir, '*'))
            gen_file = [single_file.split('/')[-1] for single_file in gen_data_list]

            # Split the data
            train_ratio = round(len(test_ct_list) * split_ratio)
            train_ct, train_lm = test_ct_list[:train_ratio], test_lm_list[:train_ratio]
            test_val_ct, test_val_lm = test_ct_list[train_ratio:], test_lm_list[train_ratio:]

            val_ratio = len(test_val_ct) // 2
            val_ct, val_lm = test_val_ct[:val_ratio], test_val_lm[:val_ratio]
            test_ct, test_lm = test_val_ct[val_ratio:], test_val_lm[val_ratio:]

            # Split the train CT into 2 (synthetic and real)
            train_ratio_2 = train_ratio // 2
            train_syn_ct = train_ct[:train_ratio_2]
            train_real_ct = train_ct[train_ratio_2:]

            # Initialize empty list
            final_train_ct = []

            # Replace with synthetic CT
            for single_ct in train_syn_ct:

                ct_data_name = single_ct.split('_')[0] + '_' + single_ct.split('_')[1] + '_' + single_ct.split('_')[2]

                if single_ct.split('_')[0] == 'LTRC':
                    ct_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4]
                else:
                    ct_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4] + '_' + \
                                single_ct.split('_')[5]

                for single_gen in gen_file:

                    gen_data_name = single_gen.split('_')[0] + '_' + single_gen.split('_')[1] + '_' + \
                                    single_gen.split('_')[2]
                    if single_gen.split('_')[0] == 'LTRC':
                        gen_series = single_gen.split('_')[3] + '_' + single_gen.split('_')[4]
                    else:
                        gen_series = single_gen.split('_')[3] + '_' + single_gen.split('_')[4] + '_' + \
                                     single_gen.split('_')[5]

                    if (gen_data_name == ct_data_name) and (gen_series == ct_series):
                        final_train_ct.append(single_gen)

            final_train_ct.extend(train_real_ct)

            # Sort the dataset
            train_lm.sort(), final_train_ct.sort()
            val_lm.sort(), val_ct.sort()
            test_lm.sort(), test_ct.sort()

            train_dataset = train_lm, final_train_ct
            val_dataset = val_lm, val_ct
            test_dataset = test_lm, test_ct

        elif self.eval_type == 'third':

            # Generate the synthetic data
            if len(glob(op.join(self.inference_data_dir, '*'))) == 0:
                super().generate_synthetic_data(save=True)
            gen_data_list = glob(op.join(self.inference_data_dir, '*'))
            gen_file = [single_file.split('/')[-1] for single_file in gen_data_list]

            # Split the data
            train_ratio = round(len(test_ct_list) * split_ratio)
            train_ct, train_lm = test_ct_list[:train_ratio], test_lm_list[:train_ratio]
            test_val_ct, test_val_lm = test_ct_list[train_ratio:], test_lm_list[train_ratio:]

            val_ratio = len(test_val_ct) // 2
            val_ct, val_lm = test_val_ct[:val_ratio], test_val_lm[:val_ratio]
            test_ct, test_lm = test_val_ct[val_ratio:], test_val_lm[val_ratio:]

            # Initialize empty list
            final_train_ct = []

            # Replace with synthetic CT
            for single_ct in train_ct:

                ct_data_name = single_ct.split('_')[0] + '_' + single_ct.split('_')[1] + '_' + single_ct.split('_')[2]

                if single_ct.split('_')[0] == 'LTRC':
                    ct_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4]
                else:
                    ct_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4] + '_' + \
                                single_ct.split('_')[5]

                for single_gen in gen_file:

                    gen_data_name = single_gen.split('_')[0] + '_' + single_gen.split('_')[1] + '_' + \
                                    single_gen.split('_')[2]
                    if single_gen.split('_')[0] == 'LTRC':
                        gen_series = single_gen.split('_')[3] + '_' + single_gen.split('_')[4]
                    else:
                        gen_series = single_gen.split('_')[3] + '_' + single_gen.split('_')[4] + '_' + \
                                     single_gen.split('_')[5]

                    if (gen_data_name == ct_data_name) and (gen_series == ct_series):
                        final_train_ct.append(single_gen)

            # Sort the dataset
            train_lm.sort(), final_train_ct.sort()
            val_lm.sort(), val_ct.sort()
            test_lm.sort(), test_ct.sort()

            train_dataset = train_lm, final_train_ct
            val_dataset = val_lm, val_ct
            test_dataset = test_lm, test_ct
        return train_dataset, val_dataset, test_dataset

    def lungs_binary_map(self, ct_data, seg_data):
        """Creates the lungs binary map for either binary or multi-class segmentation.

        :return: lungs_maps
        """
        # Load the original CT data
        ori_ct_data = self.load_original_ct(ct_data)

        # Copy 4 sets for different conditions and segment the lungs
        emphysema = ori_ct_data.copy()
        emphysema[seg_data == seg_data.min()] = emphysema.min()
        ventilated = ori_ct_data.copy()
        ventilated[seg_data == seg_data.min()] = ventilated.min()
        poorly_vent = ori_ct_data.copy()
        poorly_vent[seg_data == seg_data.min()] = poorly_vent.min()
        atelectatic = ori_ct_data.copy()
        atelectatic[seg_data == seg_data.min()] = atelectatic.min()

        # Initialize the parameters for lungs segmentation
        emphysema_range = (-1024, -900)
        ventilated_range = (-900, -500)
        poorly_vent_range = (-500, -100)
        atelectatic_range = (-100, 100)

        # Segmentation for emphysema
        emphysema[emphysema < emphysema_range[0]] = emphysema.min()
        emphysema[emphysema > emphysema_range[1]] = emphysema.min()
        emphysema[emphysema != emphysema.min()] = 1
        emphysema[emphysema == emphysema.min()] = 0

        # Segmentation for ventilated lungs
        ventilated[ventilated < ventilated_range[0]] = ventilated.min()
        ventilated[ventilated > ventilated_range[1]] = ventilated.min()
        ventilated[ventilated != ventilated.min()] = 1
        ventilated[ventilated == ventilated.min()] = 0

        # Segmentation for poorly ventilated lungs
        poorly_vent[poorly_vent < poorly_vent_range[0]] = poorly_vent.min()
        poorly_vent[poorly_vent > poorly_vent_range[1]] = poorly_vent.min()
        poorly_vent[poorly_vent != poorly_vent.min()] = 1
        poorly_vent[poorly_vent == poorly_vent.min()] = 0

        # Segmentation for atelectatic
        atelectatic[atelectatic < atelectatic_range[0]] = atelectatic.min()
        atelectatic[atelectatic > atelectatic_range[1]] = 1
        atelectatic[atelectatic != atelectatic.min()] = 1
        atelectatic[atelectatic == atelectatic.min()] = 0

        lungs_maps = (emphysema * 1) + (ventilated * 2) + (poorly_vent * 3) + (atelectatic * 4)
        if self.eval_cls == 'binary':
            lungs_maps = np.clip(lungs_maps, a_min=lungs_maps.min(), a_max=1)
        elif self.eval_cls == 'multi':
            lungs_maps = np.clip(lungs_maps, a_min=lungs_maps.min(), a_max=4)
        return lungs_maps

    def load_ct(self, ct_series):
        """Loads and resize the single CT series.

        :param ct_series: Single CT series
        :return: img_resized
        """
        data_type = ct_series.split('_')[-1].split('.')[0]

        # Load the data
        if data_type == 'inference':
            img_inf = nib.load(op.join(self.inference_data_dir, ct_series))
            single_inf_series = img_inf.get_fdata()

            affine = img_inf.affine

            # Normalize the series
            single_inf_series = single_inf_series + 1
            single_inf_series[single_inf_series < 0] = single_inf_series[single_inf_series < 0] * 1024
            single_inf_series[single_inf_series > 0] = single_inf_series[single_inf_series > 0] * 3071
            single_inf_series = (single_inf_series - np.mean(single_inf_series)) / np.std(single_inf_series)

            # Reshape to H, W, D
            image = np.rollaxis(single_inf_series, axis=0, start=3)
            image = np.squeeze(image, axis=-1)
        else:
            img_ct = nib.load(op.join(self.root_dir, 'Processed_data', 'target_data_2', ct_series))
            single_ct_series = img_ct.get_fdata()

            affine = img_ct.affine

            # Normalize the series
            single_ct_series_full = single_ct_series[:, :, :, 0] + single_ct_series[:, :, :, 1]

            single_ct_series_full = single_ct_series_full + 1
            single_ct_series_full[single_ct_series_full < 0] = single_ct_series_full[single_ct_series_full < 0] * 1024
            single_ct_series_full[single_ct_series_full > 0] = single_ct_series_full[single_ct_series_full > 0] * 3071
            single_ct_series_full = (single_ct_series_full - np.mean(single_ct_series_full)) / np.std(single_ct_series_full)

            # Reshape to H, W, D
            image = np.rollaxis(single_ct_series_full, axis=0, start=3)
        return image, affine

    def load_seg_data(self, label_series):
        """Loads and resize the lungs binary map.

        :param label_series: Single label map series
        :return: seg_data_resized
        """
        data_name = label_series.split('_')[0]
        pat_id = label_series.split('_')[2]

        if data_name == 'LTRC':

            series = label_series.split('_')[3] + '_' + label_series.split('_')[4]
            seg_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, '*' + series, 'segmentation', '*'))
            seg_data_map = {}

            for single_seg_file in seg_data_list:
                organs = single_seg_file.split('/')[-1].split('_')[-1].split('.')[0]
                if organs == 'lungs' or organs == 'vessels' or organs == 'airways':
                    single_seg_data = nib.load(single_seg_file).get_fdata()
                    data_shape = single_seg_data.shape
                    seg_data_map[organs] = single_seg_data

            seg_data = np.zeros(data_shape)
            for _, org_data in seg_data_map.items():
                seg_data = seg_data + org_data
                seg_data[seg_data != 0] = 1
                seg_data[seg_data == 0] = seg_data.min()

        elif data_name == 'UMM':

            series = label_series.split('_')[3] + '_' + label_series.split('_')[4] + '_' + label_series.split('_')[5]

            if series == '2000_01_01':
                seg_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, 'thx_endex', 'segmentation', '*'))

                for single_seg_file in seg_data_list:
                    if single_seg_file.split('/')[-1].split('.nii')[0].split('_')[-1] == 'lung':
                        seg_data = nib.load(single_seg_file).get_fdata()

            else:
                seg_data_list = glob(
                    op.join(self.root_dir, 'Data', data_name, pat_id, 'thx_endex_' + series, 'segmentation', '*'))

                for single_seg_file in seg_data_list:
                    if single_seg_file.split('/')[-1].split('.nii')[0].split('_')[-1] == 'lung':
                        seg_data = nib.load(single_seg_file).get_fdata()

        elif data_name == 'UKSH':

            series = label_series.split('_')[3] + '_' + label_series.split('_')[4] + '_' + label_series.split('_')[5]
            seg_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, 'segmentation', '*'))

            for single_seg_file in seg_data_list:
                if single_seg_file.split('/')[-1][:4] == 'UKSH' and single_seg_file.split('.')[0].split('_')[-1] == 'seg':
                    seg_data = nib.load(single_seg_file).get_fdata()
        return seg_data

    def load_original_ct(self, ct_series):
        """Loads the original CT series.

        :param ct_series: Single CT series
        :return: img_ct
        """
        data_name = ct_series.split('_')[0]
        pat_id = ct_series.split('_')[2]

        if data_name == 'LTRC':

            series = ct_series.split('_')[3] + '_' + ct_series.split('_')[4]
            ct_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, '*' + series, 'nifti', '*'))[0]

            img_ct = nib.load(ct_data_list).get_fdata()
            img_ct[img_ct < -1024] = -1024

        elif data_name == 'UMM':

            series = ct_series.split('_')[3] + '_' + ct_series.split('_')[4] + '_' + ct_series.split('_')[5]

            if series == '2000_01_01':
                ct_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, 'thx_endex', 'nifti', '*'))[0]

                img_ct = nib.load(ct_data_list).get_fdata()
                img_ct[img_ct < -1024] = -1024

            else:
                ct_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, 'thx_endex_' + series, 'nifti', '*'))[0]

                img_ct = nib.load(ct_data_list).get_fdata()
                img_ct[img_ct < -1024] = -1024

        elif data_name == 'UKSH':

            series = ct_series.split('_')[3] + '_' + ct_series.split('_')[4] + '_' + ct_series.split('_')[5]
            ct_data_list = glob(op.join(self.root_dir, 'Data', data_name, pat_id, 'nifti', '*'))[0]

            img_ct = nib.load(ct_data_list).get_fdata()
            img_ct[img_ct < -1024] = -1024
        return img_ct

    def process_data(self, dataset):
        """Process and saves the train, val, and test evaluation dataset.

        :param dataset: Train, val, or test dataset
        :return: None
        """
        # Unpack the data
        label_map, ct_data = dataset

        # Train the model with train dataset
        for ct_series, label_series in zip(ct_data, label_map):

            assert ct_series.split('_')[2] == label_series.split('_')[2], 'Not the same patient'

            data_name = ct_series.split('_')[0]
            condition = ct_series.split('_')[1]
            data_id = ct_series.split('_')[2]
            data_type = ct_series.split('_')[-1].split('.')[0]

            if data_name == 'LTRC':
                series = label_series.split('_')[3] + '_' + label_series.split('_')[4]
            elif data_name == 'UMM' or data_name == 'UKSH':
                series = label_series.split('_')[3] + '_' + label_series.split('_')[4] + '_' + label_series.split('_')[5]

            # Prepare the data for training
            ct_data, affine = self.load_ct(ct_series)
            seg_data = self.load_seg_data(label_series)

            label_data = self.lungs_binary_map(ct_series, seg_data)

            # Resize to (128, 128, 128)
            ct_data_resized = resize(ct_data, output_shape=(128, 128, 128), mode='constant', order=0, preserve_range=True, anti_aliasing=False)
            ct_data_resized = ct_data_resized[np.newaxis, :, :, :, np.newaxis]

            label_data_resized = resize(label_data, output_shape=(128, 128, 128), mode='constant', order=0, preserve_range=True, anti_aliasing=False)
            label_data_resized = label_data_resized[np.newaxis, :, :, :, np.newaxis]

            # Save the data
            ct_data_resized = nib.Nifti1Pair(ct_data_resized, affine)
            label_data_resized = nib.Nifti1Pair(label_data_resized, affine)

            if data_type == 'ct':
                file_name_ct = data_name + '_' + condition + '_' + data_id + '_' + series + '_ct.nii'
            elif data_type == 'inference':
                file_name_ct = data_name + '_' + condition + '_' + data_id + '_' + series + '_inference.nii'

            file_name_lm = data_name + '_' + condition + '_' + data_id + '_' + series + '_label_map.nii'

            nib.save(ct_data_resized, op.join(self.eval_data_dir, 'ct_data', file_name_ct))
            nib.save(label_data_resized, op.join(self.eval_data_dir, 'label_data', file_name_lm))
        pass

    def save_in_dir(self, train_data, val_data, test_data):
        """Save the respective dataset in save directory.

        :param train_data
        :param val_data
        :param test_data
        :return: None
        """
        print('\nProcessing train data and saving in Evaluation Data directory.\n')
        self.process_data(train_data)

        print('\nProcessing validation data and saving in Evaluation Data directory.\n')
        self.process_data(val_data)

        print('\nProcessing test data and saving in Evaluation Data directory.\n')
        self.process_data(test_data)
        pass

