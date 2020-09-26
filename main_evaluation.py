# Import the libraries
from Summary import DataSummary
import os.path as op
from Evaluation import Quantitative, MSSIM, U_net, VisualTuring
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

# Set the directories
root_dir = '/home/ubuntu/sl_root'
# root_dir = '/home/sailam/Desktop/MSNE/Thesis'
name = 'Run_17'
eval_type = 'first'
eval_cls = 'binary'

# Load the models
model_dir = op.join(root_dir, 'logs', name, 'models')
model_lungs_name = 'gen_lungs_model_epoch_5.h5'
model_organs_name = 'gen_organs_model_epoch_5.h5'

gen_lungs_model = load_model(op.join(model_dir, model_lungs_name), compile=False, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization})
gen_organs_model = load_model(op.join(model_dir, model_organs_name), compile=False, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization})

# Create the dataset
eval_obj = Quantitative(name=name, lungs_model=gen_lungs_model, organs_model=gen_organs_model, root_dir=root_dir,
                        merge_channel=False, eval_type=eval_type, eval_cls=eval_cls)
train_dataset, val_dataset, test_dataset = eval_obj.load_dataset(split_ratio=0.6)
eval_obj.save_in_dir(train_dataset, val_dataset, test_dataset)

# MSSIM
obj = MSSIM(name=name, lungs_model=gen_lungs_model, organs_model=gen_organs_model, merge_channel=False, root_dir=root_dir, use_tgt_organs=True)
obj.compute_results()

# Visual Turing
eval_obj = VisualTuring(name=name, lungs_model=gen_lungs_model, organs_model=gen_organs_model, root_dir=root_dir)
eval_obj.generate_images(num_images=100)

# Save the Train Test Summary
data_summary = DataSummary(name)
data_summary.evaluation_train_test_summary(train_dataset, test_dataset, val_dataset, eval_type, eval_cls, save=True)

# Train the model
model = U_net(name=name, eval_type=eval_type, eval_cls=eval_cls, label_wise_dice_coefficients=True, root_dir=root_dir)
model = model.create_model()
train_dataset, val_dataset, test_dataset = model.existing_data()

model.fit(train_dataset, val_dataset, epochs=25, batch_size=1)
model.evaluate(test_dataset)
