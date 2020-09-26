from Model import TensorflowModel

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout, Concatenate, Activation
from tensorflow.keras.models import load_model
import os
import os.path as op
import pandas as pd
import time
from glob import glob
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from utils import train_test_split, send_email, data_augmentation


class cGAN_HD_2(TensorflowModel):
    """Conditional GAN (pix2pix) model for Framework 1."""
    def __init__(self, name, save_root_dir=None, lambd=None):
        super().__init__(name, save_root_dir)
        # Variables to store data
        self.train_summary_dir = None
        self.model_dir = None

        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

        self.summary_writer = None
        self.set_backend = None
        self.last_epoch = 0

        self.create_save_directories()
        if lambd is None:
            self.lambd = 100
        else:
            self.lambd = lambd
        if save_root_dir is None:
            self.save_root_dir = '/home/ubuntu/sl_root/Processed_data'
        else:
            self.save_root_dir = save_root_dir

    def __repr__(self):
        return '{self.__class__.__name__}(name={self.name}, save_root_dir={self.save_root_dir}, lambd={self.lambd})'\
            .format(self=self)

    def create_save_directories(self):
        """Creates the save directories for the saved models, training summary directory.

        :return: None
        """
        # Set the name for the saved model and training summary directory
        self.model_dir = op.join('../logs', self.name, 'models')
        self.train_summary_dir = op.join('../logs', self.name, 'training_summary')

        if not op.exists(self.model_dir):
            if not op.exists(op.join('../logs', self.name)):
                if not op.exists('../logs'):
                    os.mkdir('../logs')
                os.mkdir(op.join('../logs', self.name))
            os.mkdir(self.model_dir)

        if not op.exists(self.train_summary_dir):
            if not op.exists(op.join('../logs', self.name)):
                if not op.exists('../logs'):
                    os.mkdir('../logs')
                os.mkdir(op.join('../logs', self.name))
            os.mkdir(self.train_summary_dir)
        return self

    @staticmethod
    def encoder(input_layer, num_filters, size=(4, 4), instance_norm=True):
        """Single encoder layer used in generator model.

        :param input_layer: Input layer from previous layer
        :param num_filters: Number of filters to be generator by the encoder
        :param size: Size of the filter (Default is (4, 4))
        :param instance_norm: Whether to apply instance normalization
        :return: encoder_layer
        """
        # Initialize the weights
        init = tf.random_normal_initializer(0.0, 0.02)

        # Convolutional Layer
        encoder_layer = Conv2D(filters=num_filters, kernel_size=size, strides=(2, 2), padding='same',
                               kernel_initializer=init, use_bias=False)(input_layer)

        # Batch normalization
        if instance_norm:
            encoder_layer = tfa.layers.InstanceNormalization()(encoder_layer)

        # Leaky ReLU
        encoder_layer = LeakyReLU(alpha=0.2)(encoder_layer)
        return encoder_layer

    @staticmethod
    def decoder(input_layer, skip_layer, num_filters, size=(4, 4), dropout=True):
        """This is the single decoder layer used in generator model.

        :param input_layer: Input layer from previous layer
        :param skip_layer: Skip layer to concatenate with the decoder layer
        :param num_filters: Number of filters to be generator by the decoder
        :param size: Size of the filter (Default is (4, 4))
        :param dropout: Whether to apply dropout
        :return: decoder_layer
        """
        # Initialize the weights
        init = tf.random_normal_initializer(0.0, 0.02)

        # Transpose Convolutional Layer
        decoder_layer = Conv2DTranspose(filters=num_filters, kernel_size=size, strides=(2, 2), padding='same',
                                        kernel_initializer=init, use_bias=False)(input_layer)

        # Instance normalization
        decoder_layer = tfa.layers.InstanceNormalization()(decoder_layer, training=True)

        # Dropout
        if dropout:
            decoder_layer = Dropout(0.5)(decoder_layer, training=True)

        # Merge with skip connection
        decoder_layer = Concatenate()([decoder_layer, skip_layer])

        # ReLU activation
        decoder_layer = Activation('relu')(decoder_layer)
        return decoder_layer

    def generator_model(self):
        """This is the full generator model of the cGAN HD (pix2pix) model.

        :return: generator_model
        """
        # Initialize the weights
        init = tf.random_normal_initializer(0.0, 0.02)

        # Image input
        input_img = tf.keras.Input(shape=(512, 512, 1))

        # Define encoder layers
        encoder_1 = self.encoder(input_img, num_filters=64, instance_norm=False)  # 256 x 256 x 64
        encoder_2 = self.encoder(encoder_1, num_filters=128)  # 128 x 128 x128
        encoder_3 = self.encoder(encoder_2, num_filters=256)  # 64 x 64 x 256
        encoder_4 = self.encoder(encoder_3, num_filters=256)  # 32 x 32 x 256
        encoder_5 = self.encoder(encoder_4, num_filters=512)  # 16 x 16 x 512
        encoder_6 = self.encoder(encoder_5, num_filters=512)  # 8 x 8 x 512
        encoder_7 = self.encoder(encoder_6, num_filters=512)  # 4 x 4 x 512
        encoder_8 = self.encoder(encoder_7, num_filters=512)  # 2 x 2 x 512

        # Bottleneck, no batch norm and relu
        bottleneck = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init,
                            use_bias=False)(encoder_8)  # 1 x 1 x 512
        bottleneck = Activation('relu')(bottleneck)

        # Decoder model
        decoder_1 = self.decoder(bottleneck, encoder_8, num_filters=512)  # 2 x 2 x 512
        decoder_2 = self.decoder(decoder_1, encoder_7, num_filters=512)  # 4 x 4 x 512
        decoder_3 = self.decoder(decoder_2, encoder_6, num_filters=512)  # 8 x 8 x 512
        decoder_4 = self.decoder(decoder_3, encoder_5, num_filters=512)  # 16 x 16 x 512 # Original dropout is False
        decoder_5 = self.decoder(decoder_4, encoder_4, num_filters=256, dropout=False)  # 32 x 32 x 256
        decoder_6 = self.decoder(decoder_5, encoder_3, num_filters=256, dropout=False)  # 64 x 64 x 256
        decoder_7 = self.decoder(decoder_6, encoder_2, num_filters=128, dropout=False)  # 128 x 128 x 128
        decoder_8 = self.decoder(decoder_7, encoder_1, num_filters=64, dropout=False)  # 256 x 256 x 64

        # Output
        output = Conv2DTranspose(filters=1, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                 kernel_initializer=init)(decoder_8)
        output_image = Activation('tanh')(output)

        # Define model
        generator_model = tf.keras.Model(input_img, output_image)
        return generator_model

    @staticmethod
    def discriminator_model():
        """Full discriminator of the model.

        :return: discriminator_model
        """
        # Initialize the weights
        init = tf.random_normal_initializer(0.0, 0.02)

        img_shape = (400, 400, 1)

        # Source and target image input
        source_img = tf.keras.Input(shape=img_shape)
        target_img = tf.keras.Input(shape=img_shape)

        # Concatenate images channel-wise
        src_tgt_img = Concatenate()([source_img, target_img])  # L : 400 x 400 x 1  # G: 200 x 200 x 1

        # C128
        d1 = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            src_tgt_img)  # L: 200 x 200 x 128  # G: 100 x 100 x 128  # RF: 4
        d1 = LeakyReLU(alpha=0.2)(d1)

        # C256
        d2 = Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            d1)  # G: 100 x 100 x 256  # L: 50 x 50 x 256  # RF: 10
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)

        # C512
        d3 = Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            d2)  # G: 50 x 50 x 512  # L: 25 x 25 x 512  # RF: 22
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = ZeroPadding2D()(d3)  # G: 52 x 52 x 512  # L: 27 x 27 x 512

        # Patch output
        d4 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=init)(
            d3)  # G: 50 x 50 x 1  # L: 25 x 25 x 1  # RF: 38
        output_patch = Activation('sigmoid')(d4)

        # Define model
        discriminator_model = tf.keras.Model([source_img, target_img], output_patch)
        return discriminator_model

    def generator_loss(self, disc_generated_output, gen_output, target):
        """Computes the generator loss.

        :param disc_generated_output: The output of label map generated by the discriminator model
        :param gen_output: The output generated by generator model
        :param target: The real CT image
        :return: total_gen_loss, gan_loss, l1_loss
        """
        # Compute the loss function
        loss_function = self.loss_func()

        # Generated GAN loss
        gan_loss = loss_function(tf.ones_like(disc_generated_output), disc_generated_output)

        # L1 loss
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        # Total generator loss
        total_gen_loss = gan_loss + (self.lambd * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """Computes the discriminator loss.

        :param disc_real_output: The output of real CT image generated by discriminator model
        :param disc_generated_output: The output of synthetic CT image generated by discriminator model
        :return: total_disc_loss
        """
        # Compute the loss function
        loss_function = self.loss_func()

        # Discriminator loss for real image
        real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)

        # Discriminator loss for generated image
        generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)

        # Total discriminator loss
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    @staticmethod
    def loss_func():
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def create_summary_writer(self):
        """Creates the summary writer.

        :return: self
        """
        col_names = ['Epoch', 'Step', 'Local_gen_total_loss', 'Global_gen_total_loss', 'Local_gen_gan_loss',
                     'Global_gen_gan_loss', 'Local_gen_l1_loss', 'Global_gen_l1_loss', 'Local_disc_loss',
                     'Global_disc_loss', 'All_gen_loss', 'All_disc_loss']

        self.summary_writer = pd.DataFrame(columns=col_names)
        return self

    def create_model(self):
        """Creates the generator, discriminator, and summary writer.

        :return: self
        """
        # Create the generator and discriminator
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()

        # Initialize the optimizer and backend
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.set_backend = tf.keras.backend.set_floatx('float32')

        # Create the summary writer
        self.create_summary_writer()
        print('Models are created.')
        return self

    def existing_data(self):
        """Obtains the existing train and test data.

        :return: existing_train_data, existing_test_data
        """
        # Set the directory and file name
        data_summary_dir = op.join('../logs', self.name, 'data_summary')
        file_name = 'Train_Test_Summary_generative.csv'

        # Read the csv and obtain the train data list
        df = pd.read_csv(op.join(data_summary_dir, file_name))
        train_data = df['Train Data'].dropna().values.tolist()
        test_data = df['Test Data'].dropna().values.tolist()

        train_data_list, test_data_list = [], []
        for single_train in train_data:
            data_name = single_train.split('_')[0]
            if data_name == 'LTRC':
                series = single_train.split('_')[3] + '_' + single_train.split('_')[4]
            else:
                series = single_train.split('_')[3] + '_' + single_train.split('_')[4] + '_' + single_train.split('_')[5]
            full_data_name = single_train.split('_')[0] + '_' + single_train.split('_')[1] + '_' + single_train.split('_')[2] + '_' + series
            train_data_list.append(full_data_name)

        for single_test in test_data:
            data_name = single_test.split('_')[0]
            if data_name == 'LTRC':
                series = single_test.split('_')[3] + '_' + single_test.split('_')[4]
            else:
                series = single_test.split('_')[3] + '_' + single_test.split('_')[4] + '_' + single_test.split('_')[5]
            full_data_name = single_test.split('_')[0] + '_' + single_test.split('_')[1] + '_' + single_test.split('_')[2] + '_' + series
            test_data_list.append(full_data_name)

        # Obtain the label map and CT list and file names
        label_map_list = glob(op.join(self.save_root_dir, 'source_data_2', '*'))
        ct_list = glob(op.join(self.save_root_dir, 'target_data_2', '*'))

        label_map_files = [single_file.split('/')[-1] for single_file in label_map_list]
        ct_files = [single_file.split('/')[-1] for single_file in ct_list]
        label_map_files.sort(), ct_files.sort()

        # Initialize empty list
        existing_train_lm, existing_train_ct = [], []
        existing_test_lm, existing_test_ct = [], []

        for single_lm, single_ct in zip(label_map_files, ct_files):

            ct_data_name = single_ct.split('_')[0] + '_' + single_ct.split('_')[1] + '_' + single_ct.split('_')[2]
            lm_data_name = single_lm.split('_')[0] + '_' + single_lm.split('_')[1] + '_' + single_lm.split('_')[2]

            assert ct_data_name == lm_data_name, 'Data is not the same.'

            data_name = single_ct.split('_')[0]
            if data_name == 'LTRC':
                series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4]
            else:
                series = single_ct.split('_')[3] + '_' + single_ct.split('_')[4] + '_' + single_ct.split('_')[5]
            full_data_name = single_ct.split('_')[0] + '_' + single_ct.split('_')[1] + '_' + single_ct.split('_')[2]\
                             + '_' + series

            if full_data_name in train_data_list:
                existing_train_lm.append(single_lm)
                existing_train_ct.append(single_ct)
            if full_data_name in test_data_list:
                existing_test_lm.append(single_lm)
                existing_test_ct.append(single_ct)
        existing_train_data = [existing_train_lm, existing_train_ct]
        existing_test_data = [existing_test_lm, existing_test_ct]
        return existing_train_data, existing_test_data

    def restore_session(self):
        """Restores the generator, discriminator, and summary writer.

        :return: self
        """
        self.model_dir = op.join('../logs', self.name, 'models')

        # Obtain the list for generator, local discriminator and global discriminator models
        gen_model_list = glob(op.join(self.model_dir, 'gens*'))
        disc_model_list = glob(op.join(self.model_dir, 'disc*'))

        if len(gen_model_list) == 0 or len(disc_model_list) == 0:
            raise NameError('No existing models.')
        else:
            # Obtain the latest model
            last_gen_model = sorted(gen_model_list, reverse=True)[0]
            last_disc_model = sorted(disc_model_list, reverse=True)[0]

            self.last_epoch = int(last_gen_model.split('_')[-1].split('.')[0])

            # Load the generator and discriminators
            self.generator = load_model(last_gen_model, compile=False,
                                        custom_objects={'InstanceNormalization': tfa.layers.InstanceNormalization})
            self.discriminator = load_model(last_disc_model, compile=False,
                                            custom_objects={'InstanceNormalization': tfa.layers.InstanceNormalization})

            # Create the summary writer
            self.create_summary_writer()
            print('Models are restored.')
        return self

    def train_step(self, source, target, epoch, step):
        """Performs training for single epoch.

        :param source: The source image (label map)
        :param target: The target image (real CT image)
        :param epoch: Single epoch
        :param step: Single step
        :return: self
        """
        # Initialize empty dictionary
        summary_dict = {}

        # Setup for training using GradientTape
        with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape(persistent=True) as disc_tape:

            # Generate the image
            source_img = source
            target_img = target
            gen_output = self.generator(source_img, training=True)

            # Discriminate the real CT and label map
            source_img_cropped = tf.image.crop_to_bounding_box(source_img, offset_height=56, offset_width=56, target_height=400, target_width=400)
            target_img_cropped = tf.image.crop_to_bounding_box(target_img, offset_height=56, offset_width=56, target_height=400, target_width=400)

            local_disc_real_output = self.discriminator([source_img_cropped, target_img_cropped], training=True)
            source_img_cropped_resized = tf.image.resize(source_img_cropped, size=(200, 200), method=tf.image.ResizeMethod.BILINEAR, antialias=True)
            target_img_cropped_resized = tf.image.resize(target_img_cropped, size=(200, 200), method=tf.image.ResizeMethod.BILINEAR, antialias=True)
            global_disc_real_output = self.discriminator([source_img_cropped_resized, target_img_cropped_resized], training=True)

            # Discriminate the generated CT and label map
            gen_output_cropped = tf.image.crop_to_bounding_box(gen_output, offset_height=56, offset_width=56, target_height=400, target_width=400)
            gen_output_cropped_resized = tf.image.resize(gen_output_cropped, size=(200, 200), method=tf.image.ResizeMethod.BILINEAR, antialias=True)
            local_disc_gen_output = self.discriminator([source_img_cropped, gen_output_cropped], training=True)
            global_disc_gen_output = self.discriminator([source_img_cropped_resized, gen_output_cropped_resized], training=True)

            # Compute the generator loss
            local_gen_total_loss, local_gen_gan_loss, local_gen_l1_loss = self.generator_loss(local_disc_gen_output, gen_output_cropped, target_img_cropped)
            global_gen_total_loss, global_gen_gan_loss, global_gen_l1_loss = self.generator_loss(global_disc_gen_output, gen_output_cropped_resized, target_img_cropped_resized)

            # Compute the discriminator loss
            local_disc_loss = self.discriminator_loss(local_disc_real_output, local_disc_gen_output)
            global_disc_loss = self.discriminator_loss(global_disc_real_output, global_disc_gen_output)

        # Compute the gradient with respect to the trainable parameters
        local_gen_gradients = gen_tape.gradient(local_gen_total_loss, self.generator.trainable_variables)
        global_gen_gradients = gen_tape.gradient(global_gen_total_loss, self.generator.trainable_variables)

        local_disc_gradients = disc_tape.gradient(local_disc_loss, self.discriminator.trainable_variables)
        global_disc_gradients = disc_tape.gradient(global_disc_loss, self.discriminator.trainable_variables)

        # Apply the gradient descent to the parameters
        self.generator_optimizer.apply_gradients(zip(local_gen_gradients, self.generator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(global_gen_gradients, self.generator.trainable_variables))

        self.discriminator_optimizer.apply_gradients(zip(local_disc_gradients, self.discriminator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(global_disc_gradients, self.discriminator.trainable_variables))

        # Compute the gradient with respect to the trainable parameters
        all_gen_loss = local_gen_total_loss + global_gen_total_loss
        all_disc_loss = local_disc_loss + global_disc_loss

        # Store the loss into the dictionary
        summary_dict['Epoch'] = epoch + 1
        summary_dict['Step'] = step + 1
        summary_dict['Local_gen_total_loss'] = local_gen_total_loss.numpy()
        summary_dict['Global_gen_total_loss'] = global_gen_total_loss.numpy()
        summary_dict['Local_gen_gan_loss'] = local_gen_gan_loss.numpy()
        summary_dict['Global_gen_gan_loss'] = global_gen_gan_loss.numpy()
        summary_dict['Local_gen_l1_loss'] = local_gen_l1_loss.numpy()
        summary_dict['Global_gen_l1_loss'] = global_gen_l1_loss.numpy()
        summary_dict['Local_disc_loss'] = local_disc_loss.numpy()
        summary_dict['Global_disc_loss'] = global_disc_loss.numpy()
        summary_dict['All_gen_loss'] = all_gen_loss.numpy()
        summary_dict['All_disc_loss'] = all_disc_loss.numpy()

        self.summary_writer = self.summary_writer.append(summary_dict, ignore_index=True)
        return self

    @staticmethod
    def load_dataset(split_ratio, save_root_dir):
        """Loads the postprocessed datasets from the directory.

        :param split_ratio: The ratio to split into train and test data.
        :param save_root_dir: The directory to the preprocessed data.
        :return: train_dataset, test_dataset
        """
        # Set the processed data directories
        train_ct, train_label_map, test_ct, test_label_map = train_test_split(split_ratio=split_ratio,
                                                                              save_root_dir=save_root_dir)

        train_dataset = train_label_map, train_ct
        test_dataset = test_label_map, test_ct
        return train_dataset, test_dataset

    def save_model(self, epoch):
        """Saves the models and summary.

        :param epoch: The epoch to save
        :return: self
        """
        # Set the name for the model
        gen_filename = 'gen_model_epoch_{}.h5'.format(epoch + 1)
        disc_filename = 'disc_model_epoch_{}.h5'.format(epoch + 1)
        train_summary_filename = 'train_summary_epoch_{}.csv'.format(epoch + 1)

        # Save the model and train summary
        self.generator.save(op.join(self.model_dir, gen_filename), include_optimizer=True)
        self.discriminator.save(op.join(self.model_dir, disc_filename), include_optimizer=True)
        self.summary_writer.to_csv(op.join(self.train_summary_dir, train_summary_filename))
        return self

    def fit(self, train_data, batch_size=1, epochs=100, save_model=1):
        """Trains the model.

        :param train_data: Data for training
        :param batch_size: Number of training sample (Default is 1)
        :param epochs: Number of epochs for training (Default is 100)
        :param save_model: Save the model as per the epochs (Default is 1)
        :return: self
        """
        for epoch in range(self.last_epoch, epochs):

            # Initialize the start time
            start = time.time()
            print('\nEpoch: ' + str(epoch + 1) + '\n')

            # Unpack the data
            label_map, ct_data = train_data

            # Shuffle the series data
            label_map, ct_data = shuffle(label_map, ct_data)

            # Train the model with train dataset
            for j, (label_series, ct_series) in enumerate(zip(label_map, ct_data)):

                assert ct_series.split('_')[2] == label_series.split('_')[2], 'Not the same patient'

                data_name = ct_series.split('_')[0]
                condition = ct_series.split('_')[1]
                pat_id = ct_series.split('_')[2]

                # Load the data
                single_ct_series = nib.load(op.join(self.save_root_dir, 'target_data_2', ct_series)).get_fdata()
                single_label_series = nib.load(op.join(self.save_root_dir, 'source_data_2', label_series)).get_fdata()

                # Shuffle the data in the series
                single_label_series, single_ct_series = shuffle(single_label_series, single_ct_series)

                # Scale to [-1, 1] for label series
                single_label_series[:, :, :, 0] = single_label_series[:, :, :, 0] + 1
                single_label_series[:, :, :, 1] = single_label_series[:, :, :, 1] + 1
                single_label_series = single_label_series[:, :, :, 0] + single_label_series[:, :, :, 1]
                single_label_series = single_label_series[:, :, :, np.newaxis]
                single_label_series = (single_label_series - (single_label_series.max() / 2)) / (single_label_series.max() / 2)

                # Scale to [-1, 1] for CT series
                single_ct_series[:, :, :, 0] = single_ct_series[:, :, :, 0] + 1
                single_ct_series[:, :, :, 1] = single_ct_series[:, :, :, 1] + 1
                single_ct_series = single_ct_series[:, :, :, 0] + single_ct_series[:, :, :, 1]
                single_ct_series = single_ct_series[:, :, :, np.newaxis]
                single_ct_series = (single_ct_series - (single_ct_series.max() / 2)) / (single_ct_series.max() / 2)

                print('Training on ... ' + str(data_name) + '(' + str(condition) + ')' + '-' + str(pat_id))

                # Send email on patient ID
                send_email(pat_id, info_type='pat_id')

                # Number of step
                num_steps = len(single_ct_series)

                for step in range(0, num_steps, batch_size):
                    # Select the label map and CT data
                    X1, X2 = single_label_series[step:step + batch_size], single_ct_series[step:step + batch_size]

                    X1, X2 = data_augmentation(X1, X2, batch_size)

                    # Convert to tensors
                    X1 = tf.cast(X1, tf.float32)
                    X2 = tf.cast(X2, tf.float32)

                    # Train (single step)
                    self.train_step(X1, X2, epoch, step)

            # Save the checkpoint and the model
            if (epoch + 1) % save_model == 0:
                # Save the generator model
                self.save_model(epoch)

                # Send email on epoch
                send_email(epoch + 1, info_type='epoch')

            # Create new summary writer
            self.create_summary_writer()

            # Initialize the end time
            end = time.time()
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, end - start))

        # Save the model at the end of training
        self.save_model(epoch)
        return self
