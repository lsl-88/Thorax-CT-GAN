# Import the libraries
from Summary import DataSummary
from Model import cGAN_HD_2

# Set the directories
data_dir = '/home/ubuntu/sl_root/Data'
# data_dir = '/home/sailam/Desktop/MSNE/Thesis/Data'
save_root_dir = '/home/ubuntu/sl_root/Processed_data'
# save_root_dir = '/home/sailam/Desktop/MSNE/Thesis/Processed_data'

# Create the cGAN model
name = 'Run_17'
model = cGAN_HD_2(name, save_root_dir, lambd=120)
model = model.create_model()

# Load the data
train_data, test_data = model.load_dataset(split_ratio=0.6, save_root_dir=save_root_dir)

# Save the Train Test Summary
data_summary = DataSummary(name)
data_summary.generative_train_test_summary(train_data, test_data, save=True)

# Perform the training
model.fit(train_data, batch_size=1, epochs=10, save_model=1)