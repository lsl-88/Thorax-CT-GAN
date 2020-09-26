from Plot import Plot

name = 'Run_17'
root_dir = '/home/sailam/Desktop/MSNE/Thesis'

plt_obj = Plot(name, pat_id='803357', series=None, root_dir=root_dir)  #pat0018
plt_obj.original_data(slice_num=0, save=False)
# plt_obj.processed_data(slice_num=180, save=False)
# plt_obj.multiplane_views(plot_target=True, sagittal_slice=200, axial_slice=100, coronal_slice=200, save=True)