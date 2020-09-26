from Plot import Plot

import os
import os.path as op
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.random import randint

from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import sum, mean
from skimage.transform import resize


class Eval(Plot):

	def __init__(self, name, pat_id, series, root_dir, eval_cls, eval_type, seg_model=None):
		super().__init__(name, pat_id, series, root_dir)

		self.seg_model = seg_model
		self.eval_cls = eval_cls
		self.eval_type = eval_type

		# Variables to store data
		self.image_dir = None
		self.eval_results_dir = None
		self.seg_model = None

		self.create_save_directories()

	def create_save_directories(self):
		"""Creates the save directories for the evaluation images.

		:return: None
		"""
		# Set the name for the directories
		self.image_dir = op.join('../logs', self.name, 'images')
		self.eval_results_dir = op.join(self.image_dir, 'evaluation')

		if not op.exists(self.eval_results_dir):
			if not op.exists(self.image_dir):
				os.mkdir(self.image_dir)
			os.mkdir(self.eval_results_dir)
		return self

	@staticmethod
	def dice_coefficient(y_true, y_pred, smooth=1):

		# Compute the intersection
		intersection = sum(y_true * y_pred, axis=[1, 2, 3])
		union = sum(y_true, axis=[1, 2, 3]) + sum(y_pred, axis=[1, 2, 3])
		return mean((2. * intersection + smooth) / (union + smooth), axis=0)

	def dice_coef_loss(self, y_true, y_pred):
		dice_coef_loss = 1 - self.dice_coefficient(y_true, y_pred)
		return dice_coef_loss

	def load_best_seg_model(self):
		"""Loads the best segmentation model.

		:return: seg_model
		"""
		# Set the segmentation model directory
		seg_model_dir = op.join(self.root_dir, 'logs', self.name, 'evaluation', 'models')
		seg_model_list = glob(op.join(seg_model_dir, '*'))

		# Obtain the lowest validation loss
		models_list = [single_model.split('/')[-1] for single_model in seg_model_list]
		val_loss_list = [single_model.split('_')[4] for single_model in models_list]
		lowest_val_loss = sorted(val_loss_list, reverse=False)[0]

		# Obtain the highest validation loss
		epoch_list = [single_model.split('_')[-1].split('.')[0] for single_model in models_list]
		highest_epoch = sorted(epoch_list, reverse=True)[0]

		# Obtain the best model
		for single_model in models_list:
			val_loss = single_model.split('_')[4]
			epoch = single_model.split('_')[-1].split('.')[0]
			if val_loss == lowest_val_loss and epoch == highest_epoch:
				best_model = single_model

		seg_model = load_model(best_model, compile=True, custom_objects={'dice_coef_loss': self.dice_coef_loss,
																		 'dice_coefficient': self.dice_coefficient})
		return seg_model

	def plot_single_image(self, slice_num=None, resize_img=False, save=False):
		"""Plots the inference CT image with the label map and CT image.

		:param slice_num: Specify the slice number for image plot
		:param resize_img: Specify to resize the image
		:param save: To save the image
		:return: None
		"""
		# Select the slice
		if slice_num is None:
			slice_num = randint(0, 128)
		elif slice_num > 127:
			raise ValueError('The slice number indicated is out of range.')

		# Load the segmentation model
		seg_model = self.load_best_seg_model()

		# Obtain the evaluation data list
		eval_data_dir = op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls)
		eval_ct = glob(op.join(eval_data_dir, 'ct_data', '*'))
		eval_lm = glob(op.join(eval_data_dir, 'label_data', '*'))
		eval_ct_list = [single_ct.split('/')[-1] for single_ct in eval_ct]
		eval_lm_list = [single_lm.split('/')[-1] for single_lm in eval_lm]
		eval_ct_list.sort(), eval_lm_list.sort()

		# Obtain the test data list
		data_summary_name = 'Train_Test_Summary_evaluation_{}_{}.csv'.format(self.eval_type, self.eval_cls)
		data_summary_dir = op.join(self.root_dir, 'logs', self.name, 'data_summary')
		df = pd.read_csv(op.join(data_summary_dir, data_summary_name))
		test_data_list = df['Test Data'].values.tolist()

		for single_test in test_data_list:

			data_name = single_test.split('_')[0]
			condition = single_test.split('_')[1]
			data_id = single_test.split('_')[2]

			if data_name == 'LTRC':
				series = single_test.split('_')[3] + '_' + single_test.split('_')[4]
			else:
				series = single_test.split('_')[3] + '_' + single_test.split('_')[4] + '_' + single_test.split('_')[5]

			if self.pat_id != data_id and self.series != series:
				raise NameError('Data not in test data.')

			elif self.pat_id == data_id and self.series == series:

				for single_ct, single_lm in zip(eval_ct_list, eval_lm_list):

					eval_data_name = single_ct.split('_')[0]
					eval_data_id = single_ct.split('_')[2]

					if eval_data_name == 'LTRC':
						eval_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4]
					else:
						eval_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4] + '_' + \
									  single_ct.split('_')[5]

					if self.pat_id == eval_data_id and self.series == eval_series:

						# Load the data
						img_lm = nib.load(op.join(eval_data_dir, 'label_data', single_lm))
						img_ct = nib.load(op.join(eval_data_dir, 'ct_data', single_ct))
						img_pred = seg_model.predict(x=img_ct, verbose=1)

						ct_slice = img_ct[0, :, :, slice_num, 0]
						lm_slice = img_lm[0, :, :, slice_num, 0]
						pred_slice = img_pred[0, :, :, slice_num, 1]

						if resize_img:
							ct_slice = resize(ct_slice, output_shape=(512, 512), mode='constant', order=0, preserve_range=True, anti_aliasing=False)
							lm_slice = resize(lm_slice, output_shape=(512, 512), mode='constant', order=0, preserve_range=True, anti_aliasing=False)
							pred_slice = resize(pred_slice, output_shape=(512, 512), mode='constant', order=0, preserve_range=True, anti_aliasing=False)

						plt.figure(figsize=(30, 10))
						plt.suptitle(eval_data_name + ' - ' + eval_data_id, fontsize=24, y=0.95)

						plt.subplot(1, 3, 1)
						plt.imshow(ct_slice, cmap='bone')
						plt.axis('off')
						plt.title('CT Slice', fontsize=18)

						plt.subplot(1, 3, 2)
						plt.imshow(lm_slice, cmap='bone')
						plt.axis('off')
						plt.title('Label Mask', fontsize=18)

						plt.subplot(1, 3, 3)
						plt.imshow(pred_slice, cmap='bone')
						plt.axis('off')
						plt.title('Prediction Mask', fontsize=18)

						if save:
							if data_name == 'LTRC':
								file_name = '{}_{}_{}_slice_{}_multiplane_view.png'.format(data_name, eval_data_id, eval_series, slice_num)
							else:
								file_name = '{}_{}_Series_{}_slice_{}_multiplane_view.png'.format(data_name, eval_data_id, eval_series, slice_num)

							plt.savefig(op.join(self.eval_results_dir, file_name))
						plt.show()
		pass

	def plot_multiple_images(self, num_images=None, resize_img=False, save=False):
		"""Plots the inference CT image with the label map and CT image.

		:param num_images: Specify the number of images for image plot
		:param resize_img: Specify to resize the image
		:param save: To save the image
		:return: None
		"""
		assert num_images > 1 or num_images < 128, 'Number of images specified out of bound.'

		# Select the slice
		slices_list = randint(0, 128, num_images)

		# Load the segmentation model
		seg_model = self.load_best_seg_model()

		# Obtain the evaluation data list
		eval_data_dir = op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls)
		eval_ct = glob(op.join(eval_data_dir, 'ct_data', '*'))
		eval_lm = glob(op.join(eval_data_dir, 'label_data', '*'))
		eval_ct_list = [single_ct.split('/')[-1] for single_ct in eval_ct]
		eval_lm_list = [single_lm.split('/')[-1] for single_lm in eval_lm]
		eval_ct_list.sort(), eval_lm_list.sort()

		# Obtain the test data list
		data_summary_name = 'Train_Test_Summary_evaluation_{}_{}.csv'.format(self.eval_type, self.eval_cls)
		data_summary_dir = op.join(self.root_dir, 'logs', self.name, 'data_summary')
		df = pd.read_csv(op.join(data_summary_dir, data_summary_name))
		test_data_list = df['Test Data'].values.tolist()

		for single_test in test_data_list:

			data_name = single_test.split('_')[0]
			condition = single_test.split('_')[1]
			data_id = single_test.split('_')[2]

			if data_name == 'LTRC':
				series = single_test.split('_')[3] + '_' + single_test.split('_')[4]
			else:
				series = single_test.split('_')[3] + '_' + single_test.split('_')[4] + '_' + single_test.split('_')[5]

			for single_ct, single_lm in zip(eval_ct_list, eval_lm_list):

				eval_data_name = single_ct.split('_')[0]
				eval_data_id = single_ct.split('_')[2]

				if eval_data_name == 'LTRC':
					eval_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4]
				else:
					eval_series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4] + '_' + single_ct.split('_')[5]

					if data_id == eval_data_id and series == eval_series:

						# Load the data
						img_lm = nib.load(op.join(eval_data_dir, 'label_data', single_lm))
						img_ct = nib.load(op.join(eval_data_dir, 'ct_data', single_ct))
						img_pred = seg_model.predict(x=img_ct, verbose=1)

						for single_slice in slices_list:

							ct_slice = img_ct[0, :, :, single_slice, 0]
							lm_slice = img_lm[0, :, :, single_slice, 0]
							pred_slice = img_pred[0, :, :, single_slice, 1]

							if resize_img:
								ct_slice = resize(ct_slice, output_shape=(512, 512), mode='constant', order=0,
												  preserve_range=True, anti_aliasing=False)
								lm_slice = resize(lm_slice, output_shape=(512, 512), mode='constant', order=0,
												  preserve_range=True, anti_aliasing=False)
								pred_slice = resize(pred_slice, output_shape=(512, 512), mode='constant', order=0,
													preserve_range=True, anti_aliasing=False)

							plt.figure(figsize=(30, 10))
							plt.suptitle(eval_data_name + ' - ' + eval_data_id + ' (Slice: ' + single_slice + ')', fontsize=24, y=0.95)

							plt.subplot(1, 3, 1)
							plt.imshow(ct_slice, cmap='bone')
							plt.axis('off')
							plt.title('CT Slice', fontsize=18)

							plt.subplot(1, 3, 2)
							plt.imshow(lm_slice, cmap='bone')
							plt.axis('off')
							plt.title('Label Mask', fontsize=18)

							plt.subplot(1, 3, 3)
							plt.imshow(pred_slice, cmap='bone')
							plt.axis('off')
							plt.title('Prediction Mask', fontsize=18)

						if save:
							if data_name == 'LTRC':
								file_name = '{}_{}_{}_slice_{}_multiplane_view.png'.format(data_name, eval_data_id, eval_series, single_slice)
							else:
								file_name = '{}_{}_Series_{}_slice_{}_multiplane_view.png'.format(data_name, eval_data_id, eval_series, single_slice)

							plt.savefig(op.join(self.eval_results_dir, file_name))
						plt.show()
		pass
