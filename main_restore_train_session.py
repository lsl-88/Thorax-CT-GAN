# Import the libraries
from Model import cGAN_HD_2

# Set the directories
data_dir = '/home/ubuntu/sl_root/Data'
# data_dir = '/home/sailam/Desktop/MSNE/Thesis/Data'
save_root_dir = '/home/ubuntu/sl_root/Processed_data'
# save_root_dir = '/home/sailam/Desktop/MSNE/Thesis/Processed_data'

# Create the cGAN model
name = 'Run_17'
model = cGAN_HD_2(name, save_root_dir, lambd=120)

# Restore the session
model.restore_session()
train_data = model.existing_train_data()
model.fit(train_data, batch_size=1, epochs=1000, save_model=1)
