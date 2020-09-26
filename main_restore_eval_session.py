from Evaluation import U_net

# Set the directories
root_dir = '/home/ubuntu/sl_root'
# root_dir = '/home/sailam/Desktop/MSNE/Thesis'
name = 'Run_17'
eval_type = 'first'
eval_cls = 'binary'

# Restore the session
model = U_net(name=name, eval_type=eval_type, eval_cls=eval_cls, label_wise_dice_coefficients=True, root_dir=root_dir)
model.restore_session()
train_dataset, val_dataset, test_dataset = model.existing_data()
model.evaluate(test_dataset)
