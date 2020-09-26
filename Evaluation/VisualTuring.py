from Evaluation import Evaluation

import os
import os.path as op
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class VisualTuring(Evaluation):
	"""Visual Turing test."""
	def __init__(self, name, lungs_model, organs_model, merge_channel, root_dir):
		super().__init__(name, lungs_model, organs_model, merge_channel, root_dir)

		self.eval_results_dir = None
		self.create_save_directories()

	def create_save_directories(self):
		"""Creates the save directories for evaluation results.

		:return: None
		"""
		self.eval_results_dir = op.join(self.root_dir, 'logs', self.name, 'evaluation_results', 'visual_turing')

		if not op.exists(self.eval_results_dir):
			if not op.exists(op.join(self.root_dir, 'logs', self.name, 'evaluation_results')):
				if not op.exists(op.join(self.root_dir, 'logs', self.name)):
					os.mkdir(op.join(self.root_dir, 'logs', self.name))
				os.mkdir(op.join(self.root_dir, 'logs', self.name, 'evaluation_results'))
			os.mkdir(self.eval_results_dir)
		return self

	@staticmethod
	def check_lungs(image, percentage=0.05):
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

	def generate_images(self, num_images):
		"""Generate pair image.

		:param num_images: The total number of pair images to generate.
		:return: None
		"""
		# Set the inference data directory
		processed_data_dir = op.join(self.root_dir, 'Processed_data')
		ct_data_dir = op.join(processed_data_dir, 'target_data_2')
		lm_data_dir = op.join(processed_data_dir, 'source_data_2')

		col_names = ['Image', 'Data', 'Synthetic CT']
		df = pd.DataFrame(columns=col_names)

		# Generate the data set
		test_ct_list, test_lm_list = super().obtain_data_list()

		idx = random.choices(range(len(test_lm_list)), k=num_images)
		selected_ct_list, selected_lm_list = [test_ct_list[i] for i in idx], [test_lm_list[i] for i in idx]

		# Initialize counter
		counter = 1
		result_dict = {}

		while not counter > num_images:
			for single_ct, single_lm in zip(selected_ct_list, selected_lm_list):

				data_name = single_lm.split('_')[0]
				condition = single_lm.split('_')[1]
				data_id = single_lm.split('_')[2]
				if data_name == 'LTRC':
					series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4]
				else:
					series_name = single_lm.split('_')[3] + '_' + single_lm.split('_')[4] + '_' + single_lm.split('_')[5]

				full_name = data_name + '_' + condition + '_' + data_id + '_' + series_name

				# Load the data
				img_ct = nib.load(op.join(ct_data_dir, single_ct)).get_fdata()
				img_lm = nib.load(op.join(lm_data_dir, single_lm)).get_fdata()

				# Generate random integer
				slice_idx = random.sample(range(len(img_lm)), k=1)

				# Target image
				tgt_lungs = img_ct[slice_idx, :, :, 1]
				tgt_lungs = tgt_lungs[:, :, :, np.newaxis]
				tgt_organs = img_ct[slice_idx, :, :, 0]
				tgt_organs = tgt_organs[:, :, :, np.newaxis]

				checker = self.check_lungs(tgt_lungs)

				if checker:
					if data_name == 'LTRC':
						print('\nGenerating image-{} ({}-{} ({})\n'.format(counter, data_name, data_id, series_name))
					else:
						print('\nGenerating image-{} ({}-{} ({})\n'.format(counter, data_name, data_id, series_name))

					# Generated image
					src_lungs = img_lm[slice_idx, :, :, 1]
					src_lungs = src_lungs[:, :, :, np.newaxis]
					gen_lungs = self.lungs_model.predict(src_lungs)

					gen_img = gen_lungs + tgt_organs
					tgt_img = tgt_lungs + tgt_organs

					# Set the seed
					seed_1 = random.randint(1, 11)
					seed_2 = np.random.uniform()

					plt.figure(figsize=(10, 5))

					# Left: Gen image, Right: Real image
					if (seed_1 % 2) == 0 and (seed_2 >= 0.5):
						plt.subplot(1, 2, 1)
						plt.imshow(gen_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						plt.subplot(1, 2, 2)
						plt.imshow(tgt_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						result_dict['Data'] = full_name
						result_dict['Image'] = counter
						result_dict['Synthetic CT'] = 'Left'

					# Left: Real image, Right: Gen image
					elif (seed_1 % 2) == 0 and (seed_2 < 0.5):
						plt.subplot(1, 2, 1)
						plt.imshow(tgt_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						plt.subplot(1, 2, 2)
						plt.imshow(gen_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						result_dict['Data'] = full_name
						result_dict['Image'] = counter
						result_dict['Synthetic CT'] = 'Right'

					# Both left and right: Gen image
					elif (seed_1 % 2) != 0 and (seed_2 >= 0.5):
						plt.subplot(1, 2, 1)
						plt.imshow(gen_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						plt.subplot(1, 2, 2)
						plt.imshow(gen_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						result_dict['Data'] = full_name
						result_dict['Image'] = counter
						result_dict['Synthetic CT'] = 'Both'

					# Both left and right: Real image
					elif (seed_1 % 2) != 0 and (seed_2 < 0.5):
						plt.subplot(1, 2, 1)
						plt.imshow(tgt_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						plt.subplot(1, 2, 2)
						plt.imshow(tgt_img[0, :, :, 0], cmap='bone')
						plt.axis('off')
						result_dict['Data'] = full_name
						result_dict['Image'] = counter
						result_dict['Synthetic CT'] = 'None'

					plt.savefig(op.join(self.eval_results_dir, 'img_{}.png'.format(str(counter))))
					df = df.append(result_dict, ignore_index=True)
					df.to_csv(op.join(self.eval_results_dir, 'visual_turing_results.csv'))
					counter += 1
		pass

