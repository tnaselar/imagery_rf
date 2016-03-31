import numpy as np
import pandas as pd
from scipy.io import loadmat
import nibabel as nib
import os
from glob import glob
from PIL import Image


##load a brain mask
def load_mask_from_nii(mask_nii_file):
    return nib.load(mask_nii_file).get_data()

##load betas from a mat file
def load_betas_from_mat_file(beta_mat_file):
    return loadmat(beta_mat_file)['all_runs_matrix']

##process a set of files that contain experiment information
def get_file_name_for_each_stim(row, stim_path, image_or_cue='image'):
    location_dict = {'north':0, 'west':1, 'southwest':2, 'northwest':3, 'east':4, 'south':5, 'northeast':6, 'southeast':7}
    stim_loc = os.path.join(stim_path,'imagery_%0.3d' %(int(row['run'][0])))
    stim_names = glob(os.path.join(stim_loc,'*.png'))
    stim_names = map(os.path.basename,stim_names)
    parts = map(lambda x: x.split('.'), stim_names)
    img_number=None
    for p in parts:
#         print '==='
#         print row['category']
#         print p[0]
#         print row['object_name']
#         print str(p[1]).replace(',','_')
#         print str(p[1]).replace(' ','_')
#         print '==='
        nm = str(p[1].replace(',','_'))
        nm = nm.replace(' ','_')
        if (row['category']==p[0]) & (row['object_name']==nm):
            img_name = os.path.join(os.path.join(stim_loc,'frame_files'),
                                    '.'.join([p[3],
                                             p[2],
                                             '%0.2d' %(location_dict[row['location']]),
                                             image_or_cue,
                                             'png']))
    return img_name

##load the stimulus information into a pandas dataframe
def load_stim_from_mat_file(stim_mat_file,stim_loc, image_or_cue='image'):
    df = pd.DataFrame(data = map(lambda x: x.split('/'),
                                 [c[0] for c in loadmat(stim_mat_file)['all_cur_subj_condits'][0]]),
                      columns=['run', 'category', 'object_name', 'location'])
    df['image_name'] = df.apply(lambda row: get_file_name_for_each_stim(row, stim_loc, image_or_cue=image_or_cue),axis=1)
    return df



##=====================================package glmdenoise betas other stimulus info
class imagery_rf_dataset(object):
    def __init__(self, subject, beta_data_frame, stim_data_frame, mask, order = 'F'):
        self.betas = beta_data_frame
        self.stim  = stim_data_frame
        self.mask = mask
        self.subject = subject
        self.shape = mask.shape
        self.order = order
    
    def view_betas(self,condition_indices,save_to = None, mean=False):
        beta_vol = np.zeros(self.mask.shape+(len(condition_indices),),order=self.order)
        for ii,beta in enumerate(condition_indices):
            cur = self.mask.ravel(order=self.order).copy()
            cur[cur > 0] = self.betas[:,beta]
            beta_vol[:,:,:,ii]=cur.reshape(self.shape,order=self.order)
        if mean:
            beta_vol = np.mean(beta_vol,axis=3)
        
        if save_to:
            nib.save(nib.Nifti1Image(beta_vol,affine=np.eye(4)),save_to)
        else:
            return beta_vol
    
    def get_condition_stimuli(self, condition_index,output='numpy',image_size=False):       
        try:
            img = Image.open(self.stim.loc[condition_index,'image_name']).resize(image_size)
        except:
            img = Image.open(self.stim.loc[condition_index,'image_name'])
        if output == 'numpy':
            img = np.array(img)
        return img  
    
    def get_stimuli_with(self,stim_col,col_contains):
        '''
        get_stimuli_with(stim_col,col_contains)
        returns index of all stimuli with "col_contains" in the stim_column
        example: all imagery runs = get_stimuli_with('run', 'img')
        '''
        n_stim = self.stim.shape[0]
        row_func = lambda row: col_contains in row
        return self.stim.loc[self.stim[stim_col].apply(row_func),:]
    
    def run_wise_z_score(self,in_place=False):        
        zscore = lambda row: (row - row.mean()) / row.std()
        if not in_place:
            z_score_df = self.betas.copy()
        else:
            z_score_df = self.betas
        runs = self.stim.groupby(by=['run'])
        for name,grp in runs:
            idx = self.get_stimuli_with('run',name).index           
            z_score_df.iloc[:,idx]= z_score_df.iloc[:,idx].apply(zscore,axis=1)
        if not in_place:
            return z_score_df   