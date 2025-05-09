%%writefile app.py
import streamlit as st
import pandas as pd
import seg_eval_wt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import preprocessing
import visualization
import metricst
import h5py
st.title('**Glioblastoma Segmentation** ')
st.image("/content/brain.png")
st.write('## MRI patch with Label')
image= st.file_uploader("Choose a image file", type="h5")
if image  is not None:
   option = st.radio('',('Different Tumor Regions', 'ALL Tumor Regions in One MRI'))
   st.write('You selected:', option)

   if option == 'Different Tumor Regions':
      h5f = h5py.File(image, 'r')
      h5f.keys()
      X=h5f['x']
      y=h5f['y']
      X_norm = preprocessing.standardize(X)
      X_norm_with_batch_dimension = np.expand_dims(X_norm, axis=0)
      model=tf.keras.models.load_model("/content/model-0001-0.3268.hdf5", compile=False)
      from tensorflow.keras.optimizers import Adam
      model.compile(optimizer = Adam(lr = 0.000001), loss =  metricst.FDICE, metrics=[metricst.dice_coefficient])
      patch_pred = model.predict(X_norm_with_batch_dimension)
      threshold = 0.7
      patch_pred[patch_pred > threshold] = 1.0  # tumor class
      patch_pred[patch_pred <= threshold] = 0.0 # no tumor
      st.write('**Ground Truth Regions**')
      fig, ax = plt.subplots(1, 4, figsize=[7, 7], squeeze=False)

      ax[0][0].imshow(X_norm[0, :, :,10], cmap='Greys_r')
      ax[0][0].set_yticks([])
      ax[0][0].set_xticks([])
      ax[0][0].set_title('Image',fontsize = 8)
      ax[0][1].imshow(y[:,:,10,1], cmap='Greys_r')
      ax[0][1].set_xticks([])
      ax[0][1].set_yticks([])
      ax[0][1].set_title('Necrotic Core',fontsize = 8)
      ax[0][2].imshow(y[:,:,10,2], cmap='Greys_r')
      ax[0][2].set_yticks([])
      ax[0][2].set_xticks([])
      ax[0][2].set_title('Edema',fontsize = 8)
      ax[0][3].imshow(y[:,:,10,3], cmap='Greys_r')
      ax[0][3].set_xticks([])
      ax[0][3].set_yticks([])
      ax[0][3].set_title('Enhancing Tumor',fontsize = 8)
      st.pyplot(fig)
      st.write('**Predicted regions**')
      fig1, ax = plt.subplots(1, 4, figsize=[7, 7], squeeze=False)

      ax[0][0].imshow(X_norm[0, :, :,10], cmap='Greys_r')
      ax[0][0].set_yticks([])
      ax[0][0].set_xticks([])
      ax[0][0].set_title('Image',fontsize = 8)
      ax[0][1].imshow(patch_pred[0,0, :, :, 10], cmap='Greys_r')
      ax[0][1].set_xticks([])
      ax[0][1].set_yticks([])
      ax[0][1].set_title('Necrotic Core',fontsize = 8)
      ax[0][2].imshow(patch_pred[0,1, :, :, 10] , cmap='Greys_r')
      ax[0][2].set_yticks([])
      ax[0][2].set_xticks([])
      ax[0][2].set_title('Edema',fontsize = 8)
      ax[0][3].imshow(patch_pred[0,2, :, :, 10] , cmap='Greys_r')
      ax[0][3].set_xticks([])
      ax[0][3].set_yticks([])
      ax[0][3].set_title('Enhancing Tumor',fontsize = 8)
      st.pyplot(fig1)
      st.write('**Evalution Metrics**')
      a1,b1,c1,d1,e1 = seg_eval_wt.metric_class(patch_pred[0,0, :, :, :],y[:,:,:,1]) # Accuracy, sensitivity, specificity and dice for TC
      a2,b2,c2,d2,e2 = seg_eval_wt.metric_class(patch_pred[0,1, :, :,:],y[:,:,:,2]) # Accuracy, sensitivity, specificity and dice for WT
      a3,b3,c3,d3,e3 = seg_eval_wt.metric_class(patch_pred[0,2, :, :,:],y[:,:,:,3]) # Accuracy, sensitivity, specificity and dice for ET
      acc = [a1,a2,a3]
      sens =[b1,b2,b3]
      spec =[c1,c2,c3]
      dice =[d1,d2,d3]
      hd=[e1,e2,e3]
      metrics_final= pd.DataFrame(columns = ['Necrotic Core', 'Edema', 'Enhancing Tumor'], index = ['Sensitivity','Specificity','Dice','Hd95'])

      for i, class_name in enumerate(metrics_final.columns):
          metrics_final.loc['Sensitivity', class_name] = sens[i]
          metrics_final.loc['Specificity', class_name] = spec[i]
          metrics_final.loc['Dice', class_name] = dice[i]
          metrics_final.loc['Hd95', class_name] = hd[i]

      st.dataframe(metrics_final,use_container_width=True)


   else:
     h5f = h5py.File(image, 'r')
     h5f.keys()
     X=h5f['x']
     y=h5f['y']
     X_norm = preprocessing.standardize(X)
     X_norm_with_batch_dimension = np.expand_dims(X_norm, axis=0)
     model=tf.keras.models.load_model("/content/model-0001-0.3268.hdf5", compile=False)
     from tensorflow.keras.optimizers import Adam
     model.compile(optimizer = Adam(lr = 0.000001), loss =  metricst.FDICE, metrics=[metricst.dice_coefficient])
     patch_pred = model.predict(X_norm_with_batch_dimension)
     threshold = 0.7
     patch_pred[patch_pred > threshold] = 1.0  # tumor class
     patch_pred[patch_pred <= threshold] = 0.0 # no tumor
     y_WT = y[:,:,:,1] +  y[:,:,:,2]+  y[:,:,:,3]
     patch_pred_WT = patch_pred[0,0, :, :, :] + patch_pred[0,1, :, :, :] + patch_pred[0,2, :, :, :]
     patch_pred_WT = np.where(patch_pred_WT >= 1, 1, 0)
     y_Core=y[:,:,:,1] + y[:,:,:,3]
     y_Core = np.where(y_Core >= 1, 1, 0)
     patch_pred_core =patch_pred[0,0, :, :, :] + patch_pred[0,2, :, :, :]
     patch_pred_core= np.where(patch_pred_core >= 1, 1,0)

     patch_pred_enc=patch_pred[0,2, :, :, :]
     from skimage import color, io
     Tlabel = np.zeros([160, 160, 16,4])
     label = np.zeros_like(Tlabel[:, :, :, 1:])
     Tlabel[:, :, :, 1] = patch_pred_WT
     Tlabel[:, :, :, 2] = patch_pred_core
     Tlabel[:, :, :, 3] = patch_pred_enc

     Tlabel[:, :, :, 1] = Tlabel[:, :, :, 1]
     Tlabel[:, :, :, 2] = Tlabel[:, :, :, 2]
     Tlabel[:, :, :, 3] = Tlabel[:, :, :, 3]
     label+= label+Tlabel[:, :, :,1: ]

     Glabel = np.zeros([160, 160, 16,3])
     Glabel[:, :, :, 0] = y_WT
     Glabel[:, :, :, 1] = y_Core
     Glabel[:, :, :, 2] = y[:,:,:,3]

     fig, ax = plt.subplots(1, 3, figsize=[7, 7], squeeze=False)

      # plane values
     X, Y, Z = (159, 159, 10)

     ax[0][0].imshow(X_norm[0, :, :,Z], cmap='Greys_r')
     ax[0][0].set_yticks([])
     ax[0][0].set_xticks([])
     ax[0][0].set_title('Image',fontsize = 8)

     ax[0][1].imshow(Glabel[:, :,Z],cmap="gray")
     ax[0][1].set_yticks([])
     ax[0][1].set_xticks([])
     ax[0][1].set_title('Ground Truth',fontsize = 8)

     ax[0][2].imshow(label[:, :, Z],cmap="gray")
     ax[0][2].set_yticks([])
     ax[0][2].set_xticks([])
     ax[0][2].set_title('Prediction by Model',fontsize = 8)
     st.pyplot(fig)
     st.write('**Evalution Metrics**')
     a1,b1,c1,d1,e1 = seg_eval_wt.metric_class(patch_pred_core,y_Core) # Accuracy, sensitivity, specificity and dice for TC
     a2,b2,c2,d2,e2 = seg_eval_wt.metric_class(patch_pred_WT,y_WT) # Accuracy, sensitivity, specificity and dice for WT
     a3,b3,c3,d3,e3 = seg_eval_wt.metric_class(patch_pred[0,2, :, :,:],y[:,:,:,3]) # Accuracy, sensitivity, specificity and dice for ET

     sens =[b1,b2,b3]
     spec =[c1,c2,c3]
     dice =[d1,d2,d3]
     hd=[e1,e2,e3]
     metrics_final= pd.DataFrame(columns = [' Core Tumor ', 'Whole Tumor', 'Enhancing Tumor'], index = ['Sensitivity','Specificity','Dice','Hd95'])

     for i, class_name in enumerate(metrics_final.columns):
         metrics_final.loc['Sensitivity', class_name] = sens[i]
         metrics_final.loc['Specificity', class_name] = spec[i]
         metrics_final.loc['Dice', class_name] = dice[i]
         metrics_final.loc['Hd95', class_name] = hd[i]

     st.dataframe(metrics_final,use_container_width=True)
else:
   st.write("Make sure you image is in h5 Format.")