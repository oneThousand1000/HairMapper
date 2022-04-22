import glob
import numpy as np

w_avg =  np.load('F:/remove_hair/source/styleGAN2_ada_model/stylegan2_ada/w_avg.npy')

for code_path in glob.glob('F:/DoubleChin/datasets/ffhq_data/remove_hair/real_image/code_wp_add_avg_loss/*.npy'):
    code = np.load(code_path)
    #print(code.shape,w_avg.shape)
    print(np.sum(code[:,0,]-w_avg)**2,np.sum(code[:, 1:, ] - code[:,0,]) ** 2)