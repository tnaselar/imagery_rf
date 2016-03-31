
# coding: utf-8

# ## Fwrf gabor model, 2014 imagery.rf data
# Using robust validation procedure.
# 

# In[ ]:

import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import pickle
from glob import glob
from PIL import Image
from imagery_rf_field.src.glmd_betas import *
from os.path import join
from time import time
from glob import glob
from scipy.io import loadmat
from scipy.stats import pearsonr
from hrf_fitting.src.feature_weighted_rf_models import make_rf_table,receptive_fields, model_space, prediction_menu, bigmult
from hrf_fitting.src.feature_weighted_rf_models import leave_k_out_training, split_em_up
from hrf_fitting.src.gabor_feature_dictionaries import gabor_feature_maps
from math import atan2, degrees
from IPython.display import Image as ipyImage

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline


# ### Step 0: load data

# #### find data files

# In[ ]:

saving_spot = '/media/tnaselar/Data/scratch/'

####with z-score
# beta_path = '/musc.repo/Data/jesse/imagery_RF/brainthresh99_0_05_with_Zscore/betas/'
# file_exp = '%s_all_runs_BETAs_w_zscore.mat'

####withOUT z-score
beta_path = '/musc.repo/Data/jesse/imagery_RF/brainthresh99_0_05_withOUT_Zscore/betas/'
file_exp = '%s_all_runs_BETAs_no_zs.mat'
subject = 'TN'
stim_loc = '/musc.repo/mri/7T.cmrr/Presentation/imagery.rf.7T.July.2014/'
cond_file_path = '/musc.repo/Data/jesse/imagery_RF/brainthresh99_0_05_withOUT_Zscore/condit_names/%s_condit_names.mat' %(subject)
mask_path = '/media/tnaselar/Data/imagery_rf.2014/masks/%s_mask.nii.gz' %(subject)
print '==============+++++++++========SUBJECT: %s' %(subject)


# #### load data

# In[ ]:

beta_df = load_betas_from_mat_file(join(beta_path,file_exp) %(subject))
mask = load_mask_from_nii(mask_path)


# In[ ]:

stim_df = load_stim_from_mat_file(cond_file_path,stim_loc)


# In[ ]:

mask = load_mask_from_nii(mask_path)


# In[ ]:

print 'volume dimensions: %s' %(mask.shape,)
print 'number of voxels: %d' %(np.sum(mask))


# #### instantiate simple class to gather all the data together

# In[ ]:

fMRI = imagery_rf_dataset(subject, beta_df, stim_df, mask)


# #### eyeball description of each condition

# In[ ]:

fMRI.stim.head(72)


# #### look at one of the stimuli

# In[ ]:

cond_dx = 512
img = fMRI.get_condition_stimuli(cond_dx)
plt.imshow(img)
print fMRI.stim.loc[cond_dx,'image_name']
print fMRI.stim.loc[cond_dx,'object_name']
print fMRI.stim.loc[cond_dx,'location']


# #### separate out the imagery and perception indices

# In[ ]:

idx = {}
idx['img'] = fMRI.get_stimuli_with('run','img').index
idx['pcp'] = fMRI.get_stimuli_with('run','pcp').index


print 'number of imagery stimuli: %d' %(len(idx['img']))
print 'number of perception stimuli: %d' %(len(idx['pcp']))


# ### Step 1: specify gabor features

# #### calculate deg_per_stimulus

# In[ ]:

native_stim_size = 600  ##pixels
viewing_distance = 100 ##cm
viewing_area = 38.8 ##cm

# Calculate the number of degrees that correspond to a single pixel. This will
# generally be a very small value, something like 0.03.
deg_per_px = degrees(atan2(.5*viewing_area, viewing_distance)) / (.5*native_stim_size)
print '%f pixels correspond to one degree' % (1./deg_per_px)
# Calculate the size of the stimulus in degrees
size_in_deg = native_stim_size * deg_per_px
print 'The size of the stimulus is %s pixels and %s visual degrees' % (native_stim_size, size_in_deg)



# #### design gabor wavelets

# In[ ]:

n_orientations = 4
deg_per_stimulus = 21.958
lowest_sp_freq = .114 ##cyc/deg
highest_sp_freq = 4.5
num_sp_freq = 8
pix_per_cycle = 4.#2.13333333
complex_cell = True
n_colors = 1 ##let's do grayscale first
diams_per_filter = 4
cycles_per_radius = 2.0

print 'D = total number of features = %d' %(n_orientations * num_sp_freq)


# In[ ]:

gfm = gabor_feature_maps(n_orientations,
                         deg_per_stimulus,
                         (lowest_sp_freq,highest_sp_freq,num_sp_freq),
                         pix_per_cycle=pix_per_cycle,complex_cell=complex_cell,
                         diams_per_filter = diams_per_filter,
                         cycles_per_radius = cycles_per_radius,
                         color_channels=n_colors)


# In[ ]:

gfm.gbr_table.head(9)


# In[ ]:

gfm.filter_stack.shape


# In[ ]:

o =  1##choose an orientation
plt.imshow(np.imag(gfm.filter_stack[o,0,:,:]), cmap='gray')


# ### Step 2: Design receptive field grid

# In[ ]:

deg_per_radius = (1., 10., 4) ##rf sizes in degrees (smallest, largest, number of sizes)
spacing = 2. ##spacing between rf's in degrees
rf = receptive_fields(deg_per_stimulus,deg_per_radius,spacing)


# In[ ]:

rf.rf_table.deg_per_radius.unique()


# In[ ]:

print 'G = number of rf models = %d' %(rf.rf_table.shape[0])


# ### Step 3: Construct a model space

# #### specify activation function

# In[ ]:

def log_act_func(x):
    return np.log(1+np.sqrt(x))


# #### instantiate a model space object

# In[ ]:

##just read in one image for now to create the feature dictionary that will initiate the model_space object
if n_colors < 3:
    ##np.newaxis,np.newaxis = time,color
    init_image = np.array(fMRI.get_condition_stimuli(0,output = 'PIL').convert('L'))[np.newaxis,np.newaxis] 
else:
    init_image = fMRI.get_condition_stimuli(0)
    init_image = np.rollaxis(init_image,axis=2,start=0)[np.newaxis] ##<<newaxis for time only

init_feature_dict = gfm.create_feature_maps(init_image)


# In[ ]:

init_image.shape


# In[ ]:

ms = {}
ms['pcp'] = model_space(init_feature_dict, rf, activation_function = log_act_func)
ms['img'] = model_space(init_feature_dict, rf, activation_function = log_act_func)


# ### Step 4: load up all stimuli at the max feature resolution

# In[ ]:

max_feature_res = np.max(ms['pcp'].feature_resolutions.values())

load_stim_func = lambda dx: np.array(fMRI.get_condition_stimuli(dx,output = 'PIL',image_size=(max_feature_res,max_feature_res)).convert('L'))[np.newaxis,np.newaxis] 


# In[ ]:

load_stim_func(0).shape


# In[ ]:

stimuli = {}
stimuli['pcp'] = np.concatenate(map(load_stim_func,
                                 idx['pcp']),
                             axis=0)


# In[ ]:

plt.imshow(stimuli['pcp'][0,0,:,:], cmap='gray')


# In[ ]:

stimuli['img'] = np.concatenate(map(load_stim_func,
                                 idx['img']),
                             axis=0)


# In[ ]:

plt.imshow(stimuli['img'][0,0,:,:],cmap='gray')


# ### Step 5: train the perception and imagery models

# #### train/test split

# In[ ]:

val_frac = 0.04687  ##24
nvox = fMRI.betas.shape[0]
n_resamples = 10 ##for a total of 240 validation samples


# #### train the model!

# In[ ]:

state_list = ['img', 'pcp']

mst = {}
params = {}
pred = {}
val_cc = {}
val_idx = {}
trn_idx = {}
for state in state_list: #[pcp', 'img']:
    params[state] = {}
    print '==========================================================%s' %(state)
    ##build feature diction. overwrite across state
    feature_dict = gfm.create_feature_maps(stimuli[state])
    ##build model space tensor
    mst[state] = ms[state].construct_model_space_tensor(feature_dict, normalize=False)
    mst[state] = ms[state].normalize_model_space_tensor(mst[state],save=True)
    voxel_data = fMRI.betas[0:nvox, idx[state]].T
    val_idx[state] = split_em_up(len(idx[state]),val_frac,n_resamples)
    

    trn_idx[state], params[state] = leave_k_out_training(val_idx[state],
                                                         mst[state],
                                                         voxel_data,
                                                         initial_feature_weights='zeros',
                                                         voxel_binsize = 25000,
                                                         rf_grid_binsize=10,
                                                         learning_rate=10**(-5.),
                                                         max_iters = 75,
                                                         early_stop_fraction=0.05,
                                                         report_every = 25)
    ##generate predictions
    Tval = len(val_idx[state][0])
    pred[state] = np.zeros((Tval*n_resamples,nvox))
    val_cc[state] = np.zeros(nvox)

    for val_iter in val_idx[state].keys():
        frf = params[state][val_iter]['frf']
        ffw = params[state][val_iter]['ffw']
        
        for v in range(nvox): 
            pred[state][(val_iter*Tval):(val_iter*Tval+Tval),v] = np.squeeze(bigmult(mst[state][np.newaxis,frf[v],val_idx[state][val_iter],:],
                                           ffw[np.newaxis,:,v, np.newaxis]))
    
    total_val_idx = np.concatenate(val_idx[state].values()).astype('int')    
    for v in range(nvox): 
        val_cc[state][v] = np.nan_to_num(pearsonr(voxel_data[total_val_idx,v],pred[state][:,v])[0])


# #### save it

# In[ ]:

saving_place = '/media/tnaselar/Data/imagery_rf.2014/model_runs/'

for ii,state in enumerate(state_list):
    saving_file = 'model_space_'+'fwrf_gabor_robust_'+state+'_'+subject+'.p'
    ms[state].params = params[state]
    ms[state].val_cc = val_cc[state]
#     ms[state].activation_function = act_func
    pickle.dump(ms[state], open( join(saving_place, saving_file), "wb"))


# ### Step 6: analysis

# #### loss history

# In[ ]:

skip = 10
for ii,state in enumerate(state_list):
    beh = params[state][0]['beh']
    plt.subplot(1,2,ii+1)
    plt.title(state)
    diff = beh[:,slice(0,-1,skip)]-beh[0,slice(0,-1,skip)]
    _=plt.plot(diff)
    plt.ylim([np.min(diff.ravel()),np.max(diff.ravel())])
plt.tight_layout()


# In[ ]:

for ii,state in enumerate(state_list):
    beh = params[state][0]['beh']
    plt.subplot(1,2,ii+1)
    plt.title(state)
    _=plt.hist(np.nan_to_num(beh[0,:]-np.min(beh,axis=0)),100)
    plt.yscale('log')


# #### rf's

# In[ ]:

for ii,state in enumerate(state_list):
    frf = params[state][0]['frf']
    plt.subplot(1,2,ii+1)
    _=plt.hist(frf,ms[state].receptive_fields.G)
    plt.xlabel('smaller-->bigger')
    plt.title(state)


# In[ ]:

for ii,state in enumerate(state_list):
    frf = params[state][0]['frf']
    plt.subplot(1,2,ii+1)
    plt.title(state)
    plt.imshow(np.sum(ms[state].receptive_fields.make_rf_stack(64, min_pix_per_radius=1)[frf,:,:], axis=0), cmap='hot')


# #### prediction accuracy histograms

# In[ ]:

####USE THIS TO LOAD UP A SAVED MST FOR VAL_CC ANALYSIS
# saving_place = '/media/tnaselar/Data/imagery_rf.2014/model_runs/'
# mst={}
# for ii,state in enumerate(['pcp','img']):
#     saving_file = 'model_space_'+'fwrf_gabor_robust_'+state+'_'+'CO'+'.p'
#     mst[state]=pickle.load(open( join(saving_place, saving_file), "r"))
    
# val_cc['pcp'] = mst['pcp'].val_cc
# val_cc['img'] = mst['img'].val_cc


# In[ ]:


count_thresh = .25
plt.figure(figsize = (10,5))
for ii,state in enumerate(state_list):#, 'img']):
    plt.subplot(2,1,ii+1)
    _=plt.hist(val_cc[state],100)
    plt.yscale('log')
    plt.ylim([10**0, 10**5])
    plt.xlim([-.5, 1.])
    plt.title(state)
    print 'number of voxels with cc > %f, %s: %d' %(count_thresh,state,np.sum(map(lambda x: x > count_thresh, val_cc[state])))


# In[ ]:

plt.figure(figsize = (10,5))
for ii,state in enumerate(state_list):
    values, base = np.histogram(val_cc[state], bins=100)
    #evaluate the cumulative
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, hold=True,label=state, linewidth=3)#     plt.yscale('log')
#     plt.ylim([10**0, 10**4])
#     plt.xlim([-1, 1.])
    
plt.legend(loc='best')    


# #### prediction accuracy head-to-head

# In[ ]:

if ('pcp' in state_list) & ('img' in state_list):
    plt.figure(figsize=(10,10))
    rng = np.linspace(-.6, .6, num=50)
    plt.plot(val_cc['pcp'], val_cc['img'], '.');
    plt.ylabel('img')
    plt.xlabel('pcp')
    plt.plot(rng, rng)
    plt.axes().set_aspect(1)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])


# #### prediction accuracy volumes

# In[ ]:

def view_vol_data(data_object, data_vol,save_to = None):
    cur = data_object.mask.ravel(order=data_object.order).copy()
    cur[cur > 0] = data_vol
    view_vol = cur.reshape(data_object.shape,order=data_object.order)
    if save_to:
        nib.save(nib.Nifti1Image(view_vol,affine=np.eye(4)),save_to)
    else:
        return view_vol


# In[ ]:

for state in state_list:
    view_vol_data(fMRI,val_cc[state], save_to = '/media/tnaselar/Data/scratch/'+state+'_gabor_val_cc_'+subject )


# In[ ]:

# ipyImage(filename='/home/tnaselar/Dropbox/Manuscripts/imagery.receptive.fields/First.try.pcp.val_cc.TN.png')


# In[ ]:

# ipyImage(filename='/home/tnaselar/Dropbox/Manuscripts/imagery.receptive.fields/First.imagery.val_cc.TN.png')

