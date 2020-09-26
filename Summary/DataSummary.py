import os
import os.path as op

import nibabel as nib
import pandas as pd


class DataSummary:
    """Summary of the preprocess, process and train test data."""
    def __init__(self, name):
        self.name = name
        self.save_dir = None
        self.current_dir = op.join(os.getcwd(), 'Summary')
        self.create_save_directories()

    def create_save_directories(self):

        # Set the name for the save directory
        self.save_dir = op.join('../logs', self.name, 'data_summary')

        if not op.exists(self.save_dir):
            if not op.exists(op.join('../logs', self.name)):
                if not op.exists('../logs'):
                    os.mkdir('../logs')
                os.mkdir(op.join('../logs', self.name))
            os.mkdir(self.save_dir)
        return self

    def LTRC_summary_processor(self):
        """Check the LTRC_summary and determines the patients to be used for training. Called in
        LTRC_healthy_preprocess function.

        :return: patient_list (List)
        """
        # Load the LTRC summary
        df = pd.read_csv(op.join(self.current_dir, 'LTRC_main_summary.csv'))

        # Place the column into a list
        col_list = list(df.columns)
        cond_list = []

        # Filter out the conditions into a list
        for i in range(len(col_list)):
            if col_list[i][0:4] == 'FIND' and col_list[i][4:6] != 'GR' and col_list[i][4:6] != 'GL':
                cond_list.append(col_list[i])

        # Filter based on CT acceptability and conditions
        df = df.loc[df['CTACCEPT'] == 1]
        for i in range(len(cond_list)):
            df = df.loc[df[cond_list[i]] == 0]

        patient_list = list(df['PATID'])
        patient_list = [str(i) for i in patient_list]
        return patient_list

    def LTRC_ARDS_summary(self):
        """Reads the csv of LTRC patient list with ARDS.

        :return: patient_list (with ARDS)
        """
        # Initialize empty list
        patient_list = []

        # Read the LTRC with ARDS csv
        df = pd.read_csv(op.join(self.current_dir, 'LTRC_ARDS_list.csv'))
        data_list = df['Unnamed: 0']

        for i, single_file in enumerate(data_list):
            # Select the even row only
            if i % 2 == 0:
                patient_list.append(single_file.split('/')[2])
        return patient_list

    def LTRC_healthy_preprocess(self, data_dir=None, save=True):
        """Obtains the data summary of the LTRC healthy dataset.

        :param data_dir: The directory of the LTRC data
        :param save: To save the csv
        :return: df (Pandas dataframe)
        """
        # Set the data directory
        if data_dir is None:
            data_dir = '/home/ubuntu/sl_root/Data/LTRC'
        else:
            data_dir = op.join(data_dir, 'LTRC')

        # Obtain the patient list from LTRC_summary_processor function
        pat_list = self.LTRC_summary_processor()

        # Ensure patient list is available in the repository
        final_pat_list = []
        avail_pat_list = os.listdir(data_dir)
        for i in avail_pat_list:
            for j in pat_list:
                if i == j:
                    final_pat_list.append(i)

        print('Generating LTRC (Healthy) Preprocess Data Summary on ' + str(len(final_pat_list)) + ' Patients ... \n')
        # Obtain the total number of patients in LTRC
        total_pat = len(final_pat_list) + 0.50 * len(final_pat_list)

        # Set the column list
        columns_list = ['Patient ID', 'Series', 'No. of Series', 'Scan Type', 'Slices']

        # Set an empty list
        df = pd.DataFrame('', index=range(int(total_pat)), columns=columns_list)

        # Initialize the counter
        j = 0

        for i in range(len(final_pat_list)):

            print('Processing Patient ID: ' + '(' + str(i + 1) + ') - ' + str(final_pat_list[i]))
            # Set the dataset directory
            dataset_dir = op.join(data_dir, final_pat_list[i])

            for single_data in os.listdir(dataset_dir):

                df['Patient ID'][j] = single_data.split('_')[0]
                df['No. of Series'][j] = len(os.listdir(dataset_dir))
                df['Series'][j] = single_data.split('_')[2] + '_' + single_data.split('_')[3]

                # Set the series directory
                series_dir = op.join(dataset_dir, single_data, 'nifti')

                for single_series in os.listdir(series_dir):

                    # Set the dataset directory
                    img = nib.load(op.join(series_dir, single_series)).get_fdata()

                    # Check the scan type
                    if img.min() < -1024:
                        scan_type = 'Circular'
                    else:
                        scan_type = 'Normal'

                    df['Scan Type'][j] = scan_type
                    df['Slices'][j] = img.shape[2]

                    # Increase the counter
                    j += 1

        # Drop empty index
        empty_index = df.index[df['Slices'] == ''].tolist()
        df.drop(empty_index, inplace=True)

        # Save the summary
        if save:
            df.to_csv(op.join(self.save_dir, 'LTRC_Healthy_preprocess_summary.csv'))
            print('\nLTRC (Healthy) Preprocess Data Summary saved.')

        # Drop if number of series more than 1
        df.drop(df.index[df['No. of Series'] > 1], inplace=True)

        print('\nLTRC (Healthy) Preprocess Data Summary generated.\n')
        return df

    def LTRC_ARDS_preprocess(self, data_dir=None, save=True):
        """Obtains the data summary of the LTRC ARDS dataset.

        :param data_dir: The directory of the LTRC data
        :param save: To save the csv
        :return: df (Pandas dataframe)
        """
        # Set the data directory
        if data_dir is None:
            data_dir = '/home/ubuntu/sl_root/Data/LTRC'
        else:
            data_dir = op.join(data_dir, 'LTRC')

        # Obtain the patient list with ARDS for LTRC
        pat_list = self.LTRC_ARDS_summary()

        # Ensure patient list is available in the repository
        final_pat_list = []
        avail_pat_list = os.listdir(data_dir)
        for i in avail_pat_list:
            for j in pat_list:
                if i == j:
                    final_pat_list.append(i)

        print('Generating LTRC (ARDS) Preprocess Data Summary on ' + str(len(final_pat_list)) + ' Patients ... \n')
        # Obtain the total number of patients in LTRC with ARDS
        total_pat = len(final_pat_list) + 0.50 * len(final_pat_list)

        # Set the column list
        columns_list = ['Patient ID', 'Series', 'No. of Series', 'Scan Type', 'Slices']

        # Set an empty list
        df = pd.DataFrame('', index=range(int(total_pat)), columns=columns_list)

        # Initialize the counter
        j = 0

        for i in range(len(final_pat_list)):

            print('Processing Patient ID: ' + '(' + str(i + 1) + ') - ' + str(final_pat_list[i]))
            # Set the dataset directory
            dataset_dir = op.join(data_dir, final_pat_list[i])

            for single_data in os.listdir(dataset_dir):

                df['Patient ID'][j] = single_data.split('_')[0]
                df['No. of Series'][j] = len(os.listdir(dataset_dir))
                df['Series'][j] = single_data.split('_')[2] + '_' + single_data.split('_')[3]

                # Set the series directory
                series_dir = op.join(dataset_dir, single_data, 'nifti')

                for single_series in os.listdir(series_dir):

                    # Set the dataset directory
                    img = nib.load(op.join(series_dir, single_series)).get_fdata()

                    # Check the scan type
                    if img.min() < -1024:
                        scan_type = 'Circular'
                    else:
                        scan_type = 'Normal'

                    df['Scan Type'][j] = scan_type
                    df['Slices'][j] = img.shape[2]

                    # Increase the counter
                    j += 1

        # Drop empty index
        empty_index = df.index[df['Slices'] == ''].tolist()
        df.drop(empty_index, inplace=True)

        # Save the summary
        if save:
            df.to_csv(op.join(self.save_dir, 'LTRC_ARDS_preprocess_summary.csv'))
            print('\nLTRC (ARDS) Preprocess Data Summary saved.')

        # Drop if number of series more than 1
        df.drop(df.index[df['No. of Series'] > 1], inplace=True)

        print('\nLTRC (ARDS) Preprocess Data Summary generated.\n')
        return df

    def UKSH_preprocess(self, data_dir=None, save=True):
        """Obtains the data summary of the UKSH dataset.

        :param data_dir: The directory of the UKSH data
        :param save: To save the csv
        :return: df (Pandas dataframe)
        """
        # Set the data directory
        if data_dir is None:
            data_dir = '/home/ubuntu/sl_root/Data/UKSH'
        else:
            data_dir = op.join(data_dir, 'UKSH')

        # Obtain the patient list
        pat_list = os.listdir(data_dir)

        print('Generating UKSH Preprocess Data Summary on ' + str(len(pat_list)) + ' Patients ... \n')
        # Obtain the total number of patients in LTRC
        total_pat = len(pat_list) + 0.20 * len(pat_list)

        # Set the column list
        columns_list = ['Patient ID', 'Series', 'No. of Series', 'Scan Type', 'Slices']

        # Set an empty list
        df = pd.DataFrame('', index=range(int(total_pat)), columns=columns_list)

        # Initialize the counter
        j = 0

        for i in range(len(pat_list)):

            print('Processing Patient ID: ' + '(' + str(i + 1) + ') - ' + str(pat_list[i]))
            # Set the dataset directory
            pat_dataset = op.join(data_dir, pat_list[i], 'nifti')

            for single_file in os.listdir(pat_dataset):
                if single_file[:4] == 'UKSH':

                    df['Series'][j] = '2000_01_01'

                    img = nib.load(op.join(pat_dataset, single_file)).get_fdata()

                    # Check the scan type
                    if img.min() < -1024:
                        scan_type = 'Circular'
                    else:
                        scan_type = 'Normal'

                    df['Patient ID'][j] = os.listdir(op.join(data_dir))[j]
                    df['Scan Type'][j] = scan_type
                    df['No. of Series'][j] = sum([file[:4] == 'UKSH' for file in os.listdir(pat_dataset)])
                    df['Slices'][j] = img.shape[2]

                    # Increase the counter
                    j += 1

        # Drop empty index
        empty_index = df.index[df['Slices'] == ''].tolist()
        df.drop(empty_index, inplace=True)

        # Save the summary
        if save:
            df.to_csv(op.join(self.save_dir, 'UKSH_preprocess_summary.csv'))
            print('\nUKSH Preprocess Data Summary saved.')

        print('\nUKSH Preprocess Data Summary generated.\n')
        return df

    def UMM_preprocess(self, data_dir=None, save=True):
        """Obtains the data summary of the UMM dataset.

        :param data_dir: The directory of the UMM data
        :param save: To save the csv
        :return: df (Pandas dataframe)
        """
        # Set the data directory
        if data_dir is None:
            data_dir = '/home/ubuntu/sl_root/Data/UMM'
        else:
            data_dir = op.join(data_dir, 'UMM')

        # Obtain the patient list
        pat_list = os.listdir(data_dir)

        print('Generating UMM Preprocess Data Summary on ' + str(len(pat_list)) + ' Patients ... \n')
        # Obtain the total number of patients in LTRC
        total_pat = len(pat_list) + 0.20 * len(pat_list)

        # Set the column list
        columns_list = ['Patient ID', 'Series', 'No. of Series', 'Scan Type', 'Slices']

        # Set an empty list
        df = pd.DataFrame('', index=range(int(total_pat)), columns=columns_list)

        # Initialize the counter
        j = 0

        for i in range(len(pat_list)):

            print('Processing Patient ID: ' + '(' + str(i + 1) + ') - ' + str(pat_list[i]))
            # Set the dataset directory
            dataset_dir = op.join(data_dir, pat_list[i])

            for file_type in os.listdir(dataset_dir):
                if file_type == 'thx_endex':

                    df['Series'][j] = '2000_01_01'
                    pat_dataset = op.join(dataset_dir, file_type, 'nifti')

                    for ct in os.listdir(pat_dataset):
                        if ct.split('_')[0] == pat_list[i]:
                            img = nib.load(op.join(pat_dataset, ct)).get_fdata()

                            # Check the scan type
                            if img.min() < -1024:
                                scan_type = 'Circular'
                            else:
                                scan_type = 'Normal'

                            df['Patient ID'][j] = ct.split('_')[0]
                            df['Scan Type'][j] = scan_type
                            df['No. of Series'][j] = len(os.listdir(dataset_dir))
                            df['Slices'][j] = img.shape[2]

                            # Increase the counter
                            j += 1

                elif file_type.split('_')[1] == 'endex':

                    year = file_type.split('_')[2]
                    month = file_type.split('_')[3]
                    day = file_type.split('_')[4]
                    date = year + '_' + month + '_' + day

                    df['Series'][j] = date
                    pat_dataset = op.join(dataset_dir, file_type, 'nifti')

                    for ct in os.listdir(pat_dataset):
                        if ct.split('_')[0] == pat_list[i]:
                            img = nib.load(op.join(pat_dataset, ct)).get_fdata()

                            # Check the scan type
                            if img.min() < -1024:
                                scan_type = 'Circular'
                            else:
                                scan_type = 'Normal'

                            df['Patient ID'][j] = ct.split('_')[0]
                            df['Scan Type'][j] = scan_type
                            df['No. of Series'][j] = sum(
                                [file.split('_')[1] == 'endex' for file in os.listdir(dataset_dir)])
                            df['Slices'][j] = img.shape[2]

                            # Increase the counter
                            j += 1

        # Drop empty index
        empty_index = df.index[df['Slices'] == ''].tolist()
        df.drop(empty_index, inplace=True)

        # Save the summary
        if save:
            df.to_csv(op.join(self.save_dir, 'UMM_preprocess_summary.csv'))
            print('\nUMM Preprocess Data Summary saved.')

        print('\nUMM Preprocess Data Summary generated.\n')
        return df

    def process_summary(self, patient_list, process_cache, dataset=None, save=True):
        """Generates the process summary.

        :param patient_list: The patient that was processed
        :param process_cache: The processed data cache
        :param dataset: The data type ('LTRC_Healthy', 'LTRC_ARDS', 'UMM' or 'UKSH')
        :param save: To save the csv
        :return: None
        """
        # Set the column name and create the data frame
        column_list = ['Patient ID', 'Date', 'Time', 'Time Taken']
        df = pd.DataFrame('', index=range(len(patient_list)), columns=column_list)

        for i in range(len(patient_list)):
            single_pat_summary = process_cache[patient_list[i]]

            df['Patient ID'][i] = patient_list[i]
            df['Date'][i] = single_pat_summary['Processed Date']
            df['Time'][i] = single_pat_summary['Processed Time']
            df['Time Taken'][i] = single_pat_summary['Time Taken']

        if save and dataset == 'LTRC_Healthy':
            df.to_csv(op.join(self.save_dir, 'LTRC_Healthy_process_summary.csv'))
        if save and dataset == 'LTRC_ARDS':
            df.to_csv(op.join(self.save_dir, 'LTRC_ARDS_process_summary.csv'))
        if save and dataset == 'UMM':
            df.to_csv(op.join(self.save_dir, 'UMM_process_summary.csv'))
        if save and dataset == 'UKSH':
            df.to_csv(op.join(self.save_dir, 'UKSH_process_summary.csv'))
        pass

    def generative_train_test_summary(self, train_data, test_data, save=True):
        """Generates the train test split summary for generative network.

        :param train_data: Nested list of train label map data and train CT data
        :param test_data: Nested list of test label map data and train CT data
        :param save: To save the csv
        :return: pass
        """
        # Set the column name and create the data frame
        pat_num = len(train_data[0]) + len(test_data[0])
        column_list = ['Train Data', 'Test Data']

        df = pd.DataFrame('', index=range(pat_num), columns=column_list)

        for i, single_file in enumerate(train_data[0]):
            df['Train Data'][i] = single_file

        for i, single_file in enumerate(test_data[0]):
            df['Test Data'][i] = single_file

        # Drop empty index
        empty_index = df.index[df['Train Data'] == ''].tolist()
        df.drop(empty_index, inplace=True)
        if save:
            df.to_csv(op.join(self.save_dir, 'Train_Test_Summary_generative.csv'))
        pass

    def evaluation_train_test_summary(self, train_data, test_data, val_data, eval_type, eval_cls, save=True):
        """Generates the train test split summary for evaluation network.

        :param train_data: Nested list of train label map data and train CT data
        :param test_data: Nested list of test label map data and test CT data
        :param val_data: Nested list of test label map data and val CT data
        :param eval_type: Specify the evaluation type (first, second or third)
        :param eval_cls: Specify the evaluation class (binary or multi)
        :param save: To save the csv
        :return: pass
        """
        assert eval_type == 'first' or eval_type == 'second' or eval_type == 'third', 'Incorrect evaluation type.'
        assert eval_cls == 'binary' or eval_cls == 'multi', 'Incorrect evaluation class.'

        # Set the column name and create the data frame
        pat_num = len(train_data[1]) + len(val_data[1]) + len(test_data[1])
        column_list = ['Train Data', 'Val Data', 'Test Data']

        df = pd.DataFrame('', index=range(pat_num), columns=column_list)

        for i, single_file in enumerate(train_data[1]):
            df['Train Data'][i] = single_file

        for i, single_file in enumerate(val_data[1]):
            df['Val Data'][i] = single_file

        for i, single_file in enumerate(test_data[1]):
            df['Test Data'][i] = single_file

        # Drop empty index
        empty_index = df.index[df['Train Data'] == ''].tolist()
        df.drop(empty_index, inplace=True)
        if save:
            file_name = 'Train_Test_Summary_evaluation_{}_{}.csv'.format(eval_type, eval_cls)
            df.to_csv(op.join(self.save_dir, file_name))
        pass

