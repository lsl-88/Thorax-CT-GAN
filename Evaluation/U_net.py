import os
import os.path as op
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import math
from glob import glob
import nibabel as nib
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import flatten, sum, mean
from functools import partial


class U_net:
    """3D segmentation network for quantitative evaluation."""
    def __init__(self, name, eval_type, eval_cls, label_wise_dice_coefficients, root_dir=None):

        if root_dir is None:
            self.root_dir = '/home/ubuntu/sl_root'
        else:
            self.root_dir = root_dir
        self.name = name
        self.image_shape = (128, 128, 128, 1)
        self.eval_type = eval_type
        self.eval_cls = eval_cls
        self.label_wise_dice_coefficients = label_wise_dice_coefficients

        self.last_epoch = 0

        # Variables to store data
        self.model = None
        self.eval_dir = None
        self.model_dir = None
        self.train_summary_dir = None
        self.create_save_directories()

    def create_save_directories(self):
        """Creates the save directories for evaluation summary and models.

        :return: None
        """
        # Set the name for the saved model and training summary directory
        self.eval_dir = op.join('../logs', self.name, 'evaluation')
        self.model_dir = op.join(self.eval_dir, 'models', self.eval_type, self.eval_cls)
        self.train_summary_dir = op.join(self.eval_dir, 'training_summary', self.eval_type, self.eval_cls)

        if not op.exists(self.model_dir):
            if not op.exists(op.join(self.eval_dir, 'models', self.eval_type)):
                if not op.exists(op.join(self.eval_dir, 'models')):
                    if not op.exists(self.eval_dir):
                        if not op.exists(op.join('../logs', self.name)):
                            if not op.exists('../logs'):
                                os.mkdir('../logs')
                            os.mkdir(op.join('../logs', self.name))
                        os.mkdir(self.eval_dir)
                    os.mkdir(op.join(self.eval_dir, 'models'))
                os.mkdir(op.join(self.eval_dir, 'models', self.eval_type))
            os.mkdir(self.model_dir)

        if not op.exists(self.train_summary_dir):
            if not op.exists(op.join(self.eval_dir, 'training_summary', self.eval_type)):
                if not op.exists(op.join(self.eval_dir, 'training_summary')):
                    if not op.exists(self.eval_dir):
                        if not op.exists(op.join('../logs', self.name)):
                            if not op.exists(op.join('../logs')):
                                os.mkdir('../logs')
                            os.mkdir(op.join('../logs', self.name))
                        os.mkdir(self.eval_dir)
                    os.mkdir(op.join(self.eval_dir, 'training_summary'))
                os.mkdir(op.join(self.eval_dir, 'training_summary', self.eval_type))
            os.mkdir(self.train_summary_dir)
        return self

    def U_net_model(self):
        """Full 3D segmentation model.

        :return: unet_model
        """
        # Image input
        inputs = tf.keras.Input(shape=self.image_shape)

        if self.eval_cls == 'binary':
            num_cls = 2
        elif self.eval_cls == 'multi':
            num_cls = 5

        encoder_1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(inputs)
        encoder_2 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(encoder_1)
        encoder_3 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(encoder_2)
        pool_1 = MaxPooling3D(pool_size=(2, 2, 2))(encoder_3)

        encoder_4 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(pool_1)
        encoder_5 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(encoder_4)
        encoder_6 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(encoder_5)
        pool_2 = MaxPooling3D(pool_size=(2, 2, 2))(encoder_6)

        encoder_7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(pool_2)
        encoder_8 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(encoder_7)
        encoder_9 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                           kernel_initializer='he_uniform')(encoder_8)
        pool_3 = MaxPooling3D(pool_size=(2, 2, 2))(encoder_9)

        encoder_10 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(pool_3)
        encoder_11 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(encoder_10)
        encoder_12 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                            kernel_initializer='he_uniform')(encoder_11)

        up_1 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same')(encoder_3)

        up_2 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same')(encoder_6)
        up_2 = tf.keras.layers.Convolution3DTranspose(filters=1, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                                      padding='same', output_padding=None)(up_2)

        up_3 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same')(encoder_9)
        up_3 = tf.keras.layers.Convolution3DTranspose(filters=1, kernel_size=(3, 3, 3), strides=(4, 4, 4),
                                                      padding='same', output_padding=None)(up_3)

        up_4 = Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same')(encoder_12)
        up_4 = tf.keras.layers.Convolution3DTranspose(filters=1, kernel_size=(3, 3, 3), strides=(8, 8, 8),
                                                      padding='same', output_padding=None)(up_4)

        concatenated = tf.keras.layers.concatenate([up_1, up_2, up_3, up_4])

        encoder_13 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same')(concatenated)
        outputs = Conv3D(filters=num_cls, kernel_size=(1, 1, 1), activation='softmax')(encoder_13)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self.label_wise_dice_coefficients:
            label_wise_dice_metrics = [self.get_label_dice_coefficient_function(index) for index in range(num_cls)]
            metrics = label_wise_dice_metrics
        else:
            metrics = [self.dice_coefficient]

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
        model.compile(loss=self.dice_coef_loss, optimizer=optimizer, metrics=metrics)
        return model

    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1.):

        # Compute the intersection
        intersection = sum(y_true * y_pred, axis=[1, 2, 3])
        union = sum(y_true, axis=[1, 2, 3]) + sum(y_pred, axis=[1, 2, 3])
        return mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def dice_coef_loss(self, y_true, y_pred):

        dice_coef_loss = 1 - self.dice_coefficient(y_true, y_pred)
        return dice_coef_loss

    def label_wise_dice_coefficient(self, y_true, y_pred, label_index):
        return self.dice_coefficient(y_true[:, label_index], y_pred[:, label_index])

    def get_label_dice_coefficient_function(self, label_index):
        f = partial(self.label_wise_dice_coefficient, label_index=label_index)
        f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
        return f

    def get_callbacks(self):
        """Creates the callback lists

        :return: callbacks list
        """
        # Initialize empty callback list
        callbacks_list = []

        # Save the model
        eval_model_filename = 'eval_model_val_loss_{val_loss:.2f}_epoch_{epoch:02d}.h5'
        checkpoint = ModelCheckpoint(op.join(self.model_dir, eval_model_filename), monitor='val_loss', verbose=0,
                                     save_best_only=True, mode='min', save_weights_only=False)
        callbacks_list.append(checkpoint)

        # Save the logs
        train_summary_filename = 'train_summary.csv'
        csv_logger = CSVLogger(op.join(self.train_summary_dir, train_summary_filename), append=True, separator=',')
        callbacks_list.append(csv_logger)

        # Reduce the LR
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
        callbacks_list.append(reduce_lr)
        return callbacks_list

    def create_model(self):
        """This function creates the U net model.

        :return: Instance to the U_net (i.e. 'self')
        """
        # Create the U net model
        self.model = self.U_net_model()

        print('Models are created.')
        return self

    def existing_data(self):
        """This function obtains the existing train, val, and test data.

        :return: existing_train_data, existing_val_data, existing_test_data
        """
        # Set the directory and file name
        data_summary_dir = op.join('../logs', self.name, 'data_summary')
        file_name = 'Train_Test_Summary_evaluation_{}_{}.csv'.format(self.eval_type, self.eval_cls)

        # Read the csv and obtain the train data list
        df = pd.read_csv(op.join(data_summary_dir, file_name))
        train_data = df['Train Data'].dropna().values.tolist()
        val_data = df['Val Data'].dropna().values.tolist()
        test_data = df['Test Data'].dropna().values.tolist()

        # Obtain the train, val, and test data list from the train test summary
        train_data_list, val_data_list, test_data_list = [], [], []
        for single_train in train_data:
            data_name = single_train.split('_')[0]
            if data_name == 'LTRC':
                series = single_train.split('_')[3] + '_' + single_train.split('_')[4]
            else:
                series = single_train.split('_')[3] + '_' + single_train.split('_')[4] + '_' + single_train.split('_')[5]
            full_data_name = single_train.split('_')[0] + '_' + single_train.split('_')[1] + '_' + single_train.split('_')[2] + '_' + series
            train_data_list.append(full_data_name)

        for single_val in val_data:
            data_name = single_val.split('_')[0]
            if data_name == 'LTRC':
                series = single_val.split('_')[3] + '_' + single_val.split('_')[4]
            else:
                series = single_val.split('_')[3] + '_' + single_val.split('_')[4] + '_' + single_val.split('_')[5]
            full_data_name = single_val.split('_')[0] + '_' + single_val.split('_')[1] + '_' + single_val.split('_')[2] + '_' + series
            val_data_list.append(full_data_name)

        for single_test in test_data:
            data_name = single_test.split('_')[0]
            if data_name == 'LTRC':
                series = single_test.split('_')[3] + '_' + single_test.split('_')[4]
            else:
                series = single_test.split('_')[3] + '_' + single_test.split('_')[4] + '_' + single_test.split('_')[5]
            full_data_name = single_test.split('_')[0] + '_' + single_test.split('_')[1] + '_' + single_test.split('_')[2] + '_' + series
            test_data_list.append(full_data_name)

        # Obtain the label map and CT list and file names
        label_map_list = glob(op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls, 'label_data', '*'))
        ct_list = glob(op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls, 'ct_data', '*'))

        label_map_files = [single_file.split('/')[-1] for single_file in label_map_list]
        ct_files = [single_file.split('/')[-1] for single_file in ct_list]
        label_map_files.sort(), ct_files.sort()

        # Initialize empty list
        existing_train_lm, existing_train_ct = [], []
        existing_val_lm, existing_val_ct = [], []
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
            if full_data_name in val_data_list:
                existing_val_lm.append(single_lm)
                existing_val_ct.append(single_ct)
            if full_data_name in test_data_list:
                existing_test_lm.append(single_lm)
                existing_test_ct.append(single_ct)
        existing_train_data = [existing_train_lm, existing_train_ct]
        existing_val_data = [existing_val_lm, existing_val_ct]
        existing_test_data = [existing_test_lm, existing_test_ct]
        return existing_train_data, existing_val_data, existing_test_data

    def restore_session(self):
        """Restores the previous session.

        :return: self
        """
        # Obtain the list for eval models
        eval_model_list = glob(op.join(self.model_dir, '*'))
        eval_model_list = [single_model.split('/')[-1] for single_model in eval_model_list]

        if len(eval_model_list) == 0:
            raise NameError('No existing models.')
        else:
            # Obtain the latest model
            min_val_loss = min([float(single_model.split('_')[4]) for single_model in eval_model_list])
            highest_epoch = max([int(single_model.split('_')[-1].split('.')[0]) for single_model in eval_model_list])
            self.last_epoch = highest_epoch

            for single_model in eval_model_list:
                val_loss = float(single_model.split('_')[4])
                epoch = int(single_model.split('_')[-1].split('.')[0])

                if val_loss == min_val_loss and epoch == highest_epoch:

                    # Load the model
                    if self.label_wise_dice_coefficients:
                        if self.eval_cls == 'binary':
                            num_cls = 2
                            label_wise_dice_metrics = [self.get_label_dice_coefficient_function(index) for index in range(num_cls)]
                            metrics = label_wise_dice_metrics
                            self.model = load_model(op.join(self.model_dir, single_model), compile=True, custom_objects={'label_0_dice_coef': metrics[0],
                                                                                                                         'label_1_dice_coef': metrics[1],
                                                                                                                         'dice_coef_loss': self.dice_coef_loss,
                                                                                                                         'dice_coefficient': self.dice_coefficient})
                        elif self.eval_cls == 'multi':
                            num_cls = 5
                            label_wise_dice_metrics = [self.get_label_dice_coefficient_function(index) for index in range(num_cls)]
                            metrics = label_wise_dice_metrics
                            self.model = load_model(op.join(self.model_dir, single_model), compile=True, custom_objects={'label_0_dice_coef': metrics[0],
                                                                                                                         'label_1_dice_coef': metrics[1],
                                                                                                                         'label_2_dice_coef': metrics[2],
                                                                                                                         'label_3_dice_coef': metrics[3],
                                                                                                                         'label_4_dice_coef': metrics[4],
                                                                                                                         'dice_coef_loss': self.dice_coef_loss,
                                                                                                                         'dice_coefficient': self.dice_coefficient})
                    elif not self.label_wise_dice_coefficients:
                        self.model = load_model(op.join(self.model_dir, single_model), compile=True, custom_objects={'dice_coef_loss': self.dice_coef_loss,
                                                                                                                     'dice_coefficient': self.dice_coefficient})
            print('Models are restored.')
        return self

    def fit(self, train_data, val_data, batch_size=1, epochs=100):
        """Trains the model using the train data.

        :param train_data: Data for training
        :param val_data: Data for validation
        :param batch_size: Specify the batch size (Default is 1)
        :param epochs: Number of epochs for training (Default is 100)
        :return: self
        """
        # Initialize the callback list
        callbacks_list = self.get_callbacks()

        # Unpack the train and validation data
        train_label, train_ct = train_data
        val_label, val_ct = val_data

        # Shuffle the series data
        train_label, train_ct = shuffle(train_label, train_ct)
        val_label, val_ct = shuffle(val_label, val_ct)

        # Initialize the train and val generator
        train_generator = DataGenerator(x_set=train_ct, y_set=train_label, batch_size=batch_size, name=self.name,
                                        eval_type=self.eval_type, eval_cls=self.eval_cls, root_dir=self.root_dir)
        val_generator = DataGenerator(x_set=val_ct, y_set=val_label, batch_size=batch_size, name=self.name,
                                      eval_type=self.eval_type, eval_cls=self.eval_cls, root_dir=self.root_dir)

        self.model.fit(x=train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks_list,
                       max_queue_size=1, use_multiprocessing=False, workers=1, steps_per_epoch=len(train_generator),
                       verbose=1)
        return self

    def evaluate(self, test_data):
        """Evaluate using the test data.

        :param test_data: Data for testing
        :return: self
        """
        # Unpack the test data
        test_label, test_ct = test_data

        # Initialize the test generator
        test_generator = DataGenerator(x_set=test_ct, y_set=test_label, batch_size=1, name=self.name,
                                       eval_type=self.eval_type, eval_cls=self.eval_cls, root_dir=self.root_dir)

        if self.eval_cls == 'binary' and not self.label_wise_dice_coefficients:
            val_loss, acc = self.model.evaluate(x=test_generator, max_queue_size=1, workers=1, verbose=1)

            print('\nVal Loss:', val_loss)
            print('Test accuracy:', acc)

        elif self.eval_cls == 'binary' and self.label_wise_dice_coefficients:
            val_loss, acc_1, acc_2 = self.model.evaluate(x=test_generator, max_queue_size=1, workers=1, verbose=1)

            print('\nVal Loss:', val_loss)
            print('Background accuracy:', acc_1)
            print('Lungs accuracy:', acc_2)

        elif self.eval_cls == 'multi' and not self.label_wise_dice_coefficients:
            val_loss, acc = self.model.evaluate(x=test_generator, max_queue_size=1, workers=1, verbose=1)

            print('\nVal Loss:', val_loss)
            print('Test accuracy:', acc)

        elif self.eval_cls == 'multi' and self.label_wise_dice_coefficients:
            val_loss, acc_1, acc_2, acc_3, acc_4, acc_5 = self.model.evaluate(x=test_generator, max_queue_size=1, workers=1, verbose=1)

            print('\nVal Loss:', val_loss)
            print('Background accuracy:', acc_1)
            print('Emphysema accuracy:', acc_2)
            print('Ventilated accuracy:', acc_3)
            print('Poorly Ventilated accuracy:', acc_4)
            print('Atelectatic accuracy:', acc_5)
        return self


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator class for evaluation."""
    def __init__(self, x_set, y_set, batch_size, name, eval_type, eval_cls, root_dir):

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

        self.name = name
        self.eval_type = eval_type
        self.eval_cls = eval_cls
        self.root_dir = root_dir

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x_batch = np.array([]).reshape(0, 128, 128, 128, 1)
        y_batch = np.array([]).reshape(0, 128, 128, 128)

        for single_x, single_y in zip(batch_x, batch_y):

            # Load the data
            X = nib.load(op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls, 'ct_data', single_x)).get_fdata()
            Y = nib.load(op.join(self.root_dir, 'Evaluation_data', self.name, self.eval_type, self.eval_cls, 'label_data', single_y)).get_fdata()
            Y = np.squeeze(Y, axis=-1)

            x_batch = np.concatenate((x_batch, X), axis=0)
            y_batch = np.concatenate((y_batch, Y), axis=0)

        if self.eval_cls == 'multi':
            X = tf.cast(x_batch, tf.float32)
            Y = tf.cast(y_batch, tf.int32)
            Y = tf.one_hot(Y, depth=5, on_value=1, off_value=0)
            Y = tf.cast(Y, tf.float32)
        elif self.eval_cls == 'binary':
            X = tf.cast(x_batch, tf.float32)
            Y = tf.cast(y_batch, tf.int32)
            Y = tf.one_hot(Y, depth=2, on_value=1, off_value=0)
            Y = tf.cast(Y, dtype=tf.float32)
        return X, Y



