# Import libraries
import os.path as op
import tensorflow_addons as tfa
from Plot import Inference
from tensorflow.keras.models import load_model


name = 'Run_17'
root_dir = '/home/ubuntu/sl_root'
# root_dir = '/home/sailam/Desktop/MSNE/Thesis'
model_dir = op.join(root_dir, 'logs', name, 'models')
model_lungs_name = 'gen_model_epoch_4.h5'
model_organs_name = 'gen_model_epoch_4.h5'

gen_lungs_model = load_model(op.join(model_dir, model_lungs_name), compile=False, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization})
gen_organs_model = load_model(op.join(model_dir, model_organs_name), compile=False, custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization})

inf_obj = Inference(name=name, model_lungs=gen_lungs_model, model_organs=gen_organs_model, pat_id='106330', series=None, merge_channel=True, root_dir=root_dir) #309938 #pat0011
inf_obj.plot_images(slice_num=150, use_original_ct=False, save=False)
# inf_obj.series_prediction(save=True)
# inf_obj.multiplane_views(sagittal_slice=375, axial_slice=25, coronal_slice=200, save=False)

