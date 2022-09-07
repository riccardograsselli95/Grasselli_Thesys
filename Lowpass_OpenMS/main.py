
"""
##############################################################################
##############################################################################
########                                                       ###############
########                   RICCARDO GRASSELLI                   ###############
########                                                       ###############
##############################################################################
##############################################################################



"""
# IMPORT
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Concatenate
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

import tensorflow.keras
import tensorflow.keras.backend as K
#

#

#from keras_layer_normalization import LayerNormalization
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ModelCheckpoint, Callback
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import zipfile 
from tensorflow.keras.callbacks import CSVLogger
#from keras.utils import plot_model

print(tf.__version__)
print(tf.keras.__version__)
import os
import json

#RN= RandomNormal(mean=0.0, stddev=0.01, seed=None)
RN= tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
### genera un tensore con una distribuzione normale



#.1) IMPORT DATA

# In[1]:


from keras.preprocessing.image import ImageDataGenerator


# In[2]:


#data_gen_args = dict(samplewise_center=True, samplewise_std_normalization=True, validation_split=0.2)
data_gen_args = dict(samplewise_center=True, rescale=1./255, validation_split=0.2)
### crea un dizionario dove ogni argomento è una keyword


# In[3]:


image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
### Il metodo ImageData Generator genera un tf.data.Dataset (un insieme di elementi gestibili con tensorflow) da dei file di immagini presenti in una directory.
### Nello specifico creo questi due generatori dandogli i parametri già impostati in data_gen_args
### **d significa "trattare le coppie chiave-valore nel dizionario come argomenti nominativi aggiuntivi a questa chiamata di funzione".


# In[4]: TRAINING SET


seed = 1
batch_size = 1
example = False
# PERCENTAGE = DATASET TYPE
# in pratica cambiando "percentage" si sceglie un DATASET diverso
# "0" corrisponde al dataset ORIGINALE
# 15, 25, 35, 45 sono i POSSIBILI DATASET CIRCOLARI 
percentage = 10
LEARNING_RATE = 0.0001
EPOCHS = 60
root_dir = ''

if example:
  path = root_dir + 'Custom_Multimodal_Unet_X4_new2/'
  path_sub = path+'Sub-sample_images/Learning_set/T2W/'   
  print('EXAMPLE DATASET')
else:
  path = root_dir + 'Dataset_open_ms/'                                    
  path_sub = path+'Sub-sample_images/Learning_set/T2W/'+str(percentage)   
  print('BIG DATASET ')                        ###  Questo metodo legge le immagini direttamente dalla directory e le aumenta 

  
# MAKE SAMPLE
# Image T2W sub
train_image_generator1 = image_datagen.flow_from_directory(           ###  Gli oggetti tf.data.Dataset possono essere scanditi con flow_from_directory.
    path_sub,
    target_size=(192,296),                                            ###  mentre il modello della rete neurale sta imparando sui dati di allenamento
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    subset='training',
    batch_size = batch_size)

# Image FLAIR
train_image_generator2 = image_datagen.flow_from_directory(
    path+'Learning_set/FLAIR',
    target_size=(192,296),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    subset='training',
    batch_size = batch_size)

# Image T2W
train_mask_generator = mask_datagen.flow_from_directory(
   path+'Learning_set/T2W',
    target_size=(192,296),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    subset='training',
    batch_size = batch_size)
#train_generator = zip(train_image_generator1,train_image_generator2)
#train_generator = zip(train_generator, train_mask_generator)

def generator_two_inputs(X1,X2):

    while True:
        X1i = X1.next()
        X2i = X2.next()
        
        yield [X1i, X2i]
#---------------------
train_generator = generator_two_inputs(train_image_generator1, train_image_generator2) ###?????
#train_generator = zip(train_generator, train_mask_generator)
train_generator = (pair for pair in zip(train_generator, train_mask_generator))
#---------------------


# In[5]: VALIDATION SET

# Image T2W sub
val_image_generator1 = image_datagen.flow_from_directory(
    path_sub,
    target_size=(192,296),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    subset='validation',
    batch_size = batch_size)
# Image FLAIR
val_image_generator2 = image_datagen.flow_from_directory(
     path+'Learning_set/FLAIR',
    target_size=(192,296),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    subset='validation',
    batch_size = batch_size)

# Image T2W
val_mask_generator = mask_datagen.flow_from_directory(
    path+'Learning_set/T2W',
    target_size=(192,296),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    subset='validation',
    batch_size = batch_size)
#---------------------
val_generator = generator_two_inputs(val_image_generator1, val_image_generator2)
#val_generator = zip(val_generator, val_mask_generator) # old version

# con questa variazione non mi da più l'errore dell 'oggetto zip durante il Model.fit
val_generator = (pair for pair in zip(val_generator, val_mask_generator))


#---------------------

# In[]: Activation
from keras.layers.advanced_activations import PReLU


class PRELU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "PRELU"
        super(PRELU, self).__init__(**kwargs)

# In[7]: Impostazioni della rete


nb_filter_in=64             # Numero di filtri iniziali nel dense block
nb_layers=5                 # numero di livelli nella dense
growth_rate=1               # Tasso di crescita
dropout_rate=0.2            # tasso di neuroni da spegnere random per l'ottimizzazione
learning_rate=1E-3          # da diminuire di un fattore 10 a metà del training
weight_decay=1E-4           # Termine di regolarizzazione dei pesi (regolarizzazione L2)
concat_axis=3 



#---------------------------------------------------------------------------------------------------------

#.3) DEFINE DENSE BLOCK

def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, PReLU 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """

    x = BatchNormalization(axis=concat_axis,
                          gamma_regularizer=l2(weight_decay),
                          beta_regularizer=l2(weight_decay))(x)
    #x = LayerNormalization()(x)
    #x=PRELU()(x)
    #x = Activation(ReLU)(x)
    x = Activation('elu')(x)  
  
    #x = KAF(int(x.get_shape()[-1]), D=5, conv=True, kernel='softplus')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x



## ARCHITETTURA DELLA RETE


########################################################################################################
############################ INPUT 2 ###################################################################

# Input layer 2
    
visible1= Input(shape=(192,296,1))
#activ0=KAF(int(visible1.get_shape()[-1]), D=5, conv=True, kernel='softplus')(visible1)

conv1 = Conv2D(64, kernel_size=(3,3), activation='elu', padding='same',use_bias=True,  kernel_initializer=RN)(visible1)
#---
conv2 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same',use_bias=False, kernel_initializer=RN)(conv1)

#---------------------------------------------------------------------------
## Dense block1

nb_filter=nb_filter_in
list_feat1 = [conv2]

for i in range(nb_layers): #numeri di livelli nel dense block (5)
        x = conv_factory(conv2, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat1.append(x)
        dense_c1 = Concatenate(axis=concat_axis)(list_feat1)
        nb_filter=nb_filter * growth_rate


#-----------------------------------------------------------------------------
# Translation layer 1

bn1=BatchNormalization(axis=-1,epsilon=1e-06, momentum=0.9, weights=None)(dense_c1)
#ln1=LayerNormalization()(dense_c1)
#activ1=KAF(int(bn1.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn1)

conv3 = Conv2D(64, (1, 1), kernel_initializer="he_uniform", padding="same",use_bias=False, kernel_regularizer=l2(weight_decay))(bn1)
                
pool1= AveragePooling2D((2, 2), strides=(2, 2))(conv3)
#---
conv4 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(pool1)
#-----------------------------------------------------------------------------

## Dense block2

nb_filter=nb_filter_in
list_feat2 = [conv4]

for i in range(nb_layers):
        x = conv_factory(conv4, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat2.append(x)
        dense_c2 = Concatenate(axis=concat_axis)(list_feat2)
        nb_filter=nb_filter * growth_rate


#-----------------------------------------------------------------------------
# Translation layer 2

bn2=BatchNormalization(axis=-1,epsilon=1e-06, momentum=0.9, weights=None)(dense_c2)
#ln1=LayerNormalization()(dense_c1)
#activ2=KAF(int(bn2.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn2)


conv5 = Conv2D(64, (1, 1), kernel_initializer="he_uniform", padding="same",use_bias=False, kernel_regularizer=l2(weight_decay))(bn2)
                
pool2= AveragePooling2D((2, 2), strides=(2, 2))(conv5)
#---
conv6 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(pool2)
#-----------------------------------------------------------------------------

## Dense block3

nb_filter=nb_filter_in
list_feat3 = [conv6]

for i in range(nb_layers):
        x = conv_factory(conv6, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat3.append(x)
        dense_c3 = Concatenate(axis=concat_axis)(list_feat3)
        nb_filter=nb_filter * growth_rate


#-----------------------------------------------------------------------------
conv7 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(dense_c3)

########################################################################################################
############################ INPUT 1 ###################################################################

# Input layer 1

visible2= Input(shape=(192,296,1))
#activ2b=KAF(int(visible2.get_shape()[-1]), D=5, conv=True, kernel='softplus')(visible2)

conv8 = Conv2D(64, kernel_size=(3,3), activation='elu', padding='same',use_bias=True,  kernel_initializer=RN)(visible2)
#---
# MERGE 1

merge1 = concatenate([conv8,conv3], axis = 3)
#---
conv9 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(merge1)

#---------------------------------------------------------------------------
## Dense block4
nb_filter=nb_filter_in

list_feat4 = [conv9]

for i in range(nb_layers):
        x = conv_factory(conv9, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat4.append(x)
        dense_c4 = Concatenate(axis=concat_axis)(list_feat4)
        nb_filter=nb_filter * growth_rate


#-----------------------------------------------------------------------------
# Translation layer 3

bn3=BatchNormalization(axis=-1,epsilon=1e-06, momentum=0.9, weights=None)(dense_c4)
#ln2=LayerNormalization()(dense_c2)
#activ3=KAF(int(bn2.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn3)
conv10 = Conv2D(64, (1, 1), kernel_initializer="he_uniform", padding="same",use_bias=False, kernel_regularizer=l2(weight_decay))(bn3)
               
pool3= AveragePooling2D((2, 2), strides=(2, 2))(conv10)

# MERGE 2
merge2 = concatenate([pool3,conv5], axis = 3)

#---
conv11 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(merge2)

#---------------------------------------------------------------------------
## Dense block5
nb_filter=nb_filter_in

list_feat5 = [conv11]

for i in range(nb_layers):
        x = conv_factory(conv11, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat5.append(x)
        dense_c5 = Concatenate(axis=concat_axis)(list_feat5)
        nb_filter=nb_filter * growth_rate


#-----------------------------------------------------------------------------


# Translation layer 4

bn4=BatchNormalization(axis=-1,epsilon=1e-06, momentum=0.9, weights=None)(dense_c5)
#ln2=LayerNormalization()(dense_c2)
#activ4=KAF(int(bn4.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn4)

conv12 = Conv2D(64, (1, 1), kernel_initializer="he_uniform", padding="same",use_bias=False, kernel_regularizer=l2(weight_decay))(bn4)
               
pool4= AveragePooling2D((2, 2), strides=(2, 2))(conv12)

# MERGE 3
merge3 = concatenate([pool4,conv7], axis = 3)

#---
conv13 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(merge3)

#-----------------------------------------------------------------------------
## Dense block6
nb_filter=nb_filter_in

list_feat6 = [conv13]

for i in range(nb_layers):
        x = conv_factory(conv13, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat6.append(x)
        dense_c6 = Concatenate(axis=concat_axis)(list_feat6)
        nb_filter=nb_filter * growth_rate

#-----------------------------------------------------------------------------
# Translation layer 5

bn5 = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(dense_c6)
#ln3=LayerNormalization()(dense_c3)
#activ5=KAF(int(bn5.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn5)
conv14 = Conv2D(64, (1, 1), kernel_initializer="he_uniform", padding="same",use_bias=False, kernel_regularizer=l2(weight_decay))(bn5)                              


pool5= AveragePooling2D((2, 2), strides=(2, 2))(conv14)
#--
conv15 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(pool5)

#------------------------------------------------------------------------------

## Dense block7
nb_filter=nb_filter_in

list_feat7 = [conv15]

for i in range(nb_layers):
        x = conv_factory(conv15, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat7.append(x)
        dense_c7 = Concatenate(axis=concat_axis)(list_feat7)
        nb_filter=nb_filter * growth_rate


#----------------------------------------------------------------------------

# Translation layer 6 (espansione)

bn6= BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(dense_c7)
#ln4=LayerNormalization()(dense_c4)
#activ6=KAF(int(bn6.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn6)

conv16 = Conv2DTranspose(64, (4, 4),strides=(2, 2), padding="same",use_bias=False,kernel_regularizer=l2(weight_decay))(bn6) 

# MERGE 4
merge4 = concatenate([conv16,conv14], axis = 3)

#--
conv17 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(merge4)

#------------------------------------------------------------------------------

## Dense block8
nb_filter=nb_filter_in

list_feat8 = [conv17]

for i in range(nb_layers):
        x = conv_factory(conv17, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat8.append(x)
        dense_c8 = Concatenate(axis=concat_axis)(list_feat8)
        nb_filter=nb_filter * growth_rate


#----------------------------------------------------------------------------


# Translation layer 7 (espansione)

bn7= BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(dense_c8)
#ln4=LayerNormalization()(dense_c4)
#activ7=KAF(int(bn7.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn7)

conv18 = Conv2DTranspose(64, (4, 4),strides=(2, 2), padding="same",use_bias=False,kernel_regularizer=l2(weight_decay))(bn7) 

# MERGE 5
merge5 = concatenate([conv18,conv12], axis = 3)

#--
conv19 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(merge5)

#------------------------------------------------------------------------------

## Dense block9
nb_filter=nb_filter_in

list_feat9 = [conv19]

for i in range(nb_layers):
        x = conv_factory(conv19, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat9.append(x)
        dense_c9 = Concatenate(axis=concat_axis)(list_feat9)
        nb_filter=nb_filter * growth_rate


#----------------------------------------------------------------------------


# Translation layer 8 (espansione)

bn8= BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(dense_c9)
#ln4=LayerNormalization()(dense_c4)
#activ8=KAF(int(bn8.get_shape()[-1]), D=5, conv=True, kernel='softplus')(bn8)

conv20 = Conv2DTranspose(64, (4, 4),strides=(2, 2), padding="same",use_bias=False,kernel_regularizer=l2(weight_decay))(bn8) 

# MERGE 6
merge6 = concatenate([conv20,conv10], axis = 3)

#--
conv21 = Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer=RN)(merge6)


#------------------------------------------------------------------------------
## Dense block 10
nb_filter=nb_filter_in

list_feat10 = [conv21]

for i in range(nb_layers):
        x = conv_factory(conv21, concat_axis, nb_filter,
                         dropout_rate, weight_decay)
        list_feat10.append(x)
        dense_c10 = Concatenate(axis=concat_axis)(list_feat10)
        nb_filter=nb_filter * growth_rate

#------------------------------------------------------------------------------
# Recostruction layer

conv22= Conv2D(1, kernel_size=(3,3), padding='same',use_bias=True,  kernel_initializer=RN) (dense_c10)


model_standard = Model(inputs = [visible1,visible2], outputs = [conv22])


# In[10]: Model summary

#model_standard.summary()

# In[11]:Metric
from skimage.metrics import structural_similarity as ssim

# In[11]:Metric (original)


# SSIM
def SSIM(y_true, y_pred):
    
    #15/9/2021 -> prima era così ed ho aggiornato il codice
    #patches_true = tf.extract_image_patches(y_true, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME") 
    #patches_pred = tf.extract_image_patches(y_pred, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    patches_true = tf.compat.v1.extract_image_patches(y_true, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME") 
    patches_pred = tf.compat.v1.extract_image_patches(y_pred, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    mean_true = K.mean(patches_true, axis=3) 
    mean_pred = K.mean(patches_pred, axis = 3) 
    mt2= mean_true**2 
    mp2= mean_pred**2 
    c1= (0.01*1)**2
    ssim1= (2*mean_true*mean_pred + c1)/(mt2+mp2+c1) 
    var_true = K.var(patches_true, axis=3) 
    var_pred = K.var(patches_pred, axis=3) 
    std_true= K.sqrt(var_true) 
    std_pred = K.sqrt(var_pred) 
    covar_true_pred = K.mean(patches_true*patches_pred, axis=-1) - mean_true*mean_pred  
    c2 = (0.03 *1)**2
    ssim2 = (2 * covar_true_pred + c2)/(var_pred + var_true + c2) 
    ssim=ssim1*ssim2 
    #dssim= 1-ssim 
    return K.mean(ssim)

def DSSIM(y_true, y_pred):
    
    #15/9/2021 -> prima era così ed ho aggiornato il codice
    #patches_true = tf.extract_image_patches(y_true, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME") 
    #patches_pred = tf.extract_image_patches(y_pred, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    patches_true = tf.compat.v1.extract_image_patches(y_true, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME") 
    patches_pred = tf.compat.v1.extract_image_patches(y_pred, [1, 8, 8, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    mean_true = K.mean(patches_true, axis=3) 
    mean_pred = K.mean(patches_pred, axis = 3) 
    mt2= mean_true**2 
    mp2= mean_pred**2 
    c1= (0.01*1)**2
    ssim1= (2*mean_true*mean_pred + c1)/(mt2+mp2+c1) 
    var_true = K.var(patches_true, axis=3) 
    var_pred = K.var(patches_pred, axis=3) 
    std_true= K.sqrt(var_true) 
    std_pred = K.sqrt(var_pred) 
    covar_true_pred = K.mean(patches_true*patches_pred, axis=-1) - mean_true*mean_pred  
    c2 = (0.03 *1)**2
    ssim2 = (2 * covar_true_pred + c2)/(var_pred + var_true + c2) 
    ssim=ssim1*ssim2 
    dssim= 1-ssim 
    return K.mean(dssim)

# In[12]: LOSS FUNCTION
#from sklearn.metrics import mean_absolute_error

#from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error

def custom_loss(y_true, y_pred):
    loss1 = mean_absolute_error(y_true, y_pred)
    loss2 = DSSIM(y_true, y_pred)
    return loss1 + loss2

# In[13]
    
# Folder Tensorboard
folder = 'Std_bigger'
savelog_path = root_dir + 'Tensorboard/log_dataset'+str(percentage)+'_LR'+str(LEARNING_RATE)+'.csv'
print('SAVELOG_PATH: ', savelog_path)
ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
TB = TensorBoard(root_dir + 'Tensorboard/', write_graph=False, update_freq=100)
csv_logger = CSVLogger(savelog_path, append=True, separator=';')



class CustomLogSavingCallback(Callback):
    def __init__(self, log_path, parameters):
        super(CustomLogSavingCallback, self).__init__()
        self.log_path = log_path
        self.parameters = parameters

    def write_results(self, history_log):
      print('HISTORY_LOG: ', history_log, ' - ', type(history_log))
      percentage = self.parameters[0]
      LEARNING_RATE = self.parameters[1]
      with open(self.log_path, "r") as file:
        results_dictionary = json.load(file)
        file.close()
        #print('Results: ', results_dictionary)
        for key in history_log.keys():
          value = history_log[key]
          try:
            len_results = len(results_dictionary[key][str(percentage)][str(LEARNING_RATE)])
            h = {str(len_results): str(value)}
            results_dictionary[key][str(percentage)][str(LEARNING_RATE)].update(h)
          except:
            try:
                len_results = '0'
                h = {str(len_results): str(value)}
                l = {str(LEARNING_RATE):h}
                results_dictionary[key][str(percentage)].update(l)
            except:
                len_results = '0'
                h = {str(len_results): str(value)}
                l = {str(LEARNING_RATE):h}
                m = {str(percentage):l}
                results_dictionary[key].update(m)

        with open(self.log_path, 'w') as file:
          json.dump(results_dictionary, file,  indent=4)
          file.close()
        final_text = 'ALL DONE!'
        return final_text
   
    def on_epoch_end(self, epoch, logs=None):
        #keys = list(logs.keys())
        log_text = self.write_results(history_log=logs)
        print("Write Log of epoch number {}")
        print("Result of writing: {}".format(epoch, log_text))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

# CheckPoint
# 
# decomment to LOAD BEST CHECKPOINT
# https://www.tensorflow.org/tutorials/keras/save_and_load

folder_checkpoint = str(percentage)
checkpoint_path = root_dir + "CheckPoint/"+str(percentage)+"/"+str(LEARNING_RATE)+"/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_path)
print(checkpoint_dir)
print(os.listdir(checkpoint_dir))
if len(os.listdir(checkpoint_dir)) > 0:
  load_checkpoints = True
  last_checkpoint = os.listdir(checkpoint_dir)
  max = 0
  for i, name in enumerate(last_checkpoint):
    if name != 'checkpoint':
      number = int(str(name[5]) + str(name[6]))
      if number > max:
        max = number
        checkfile_name = last_checkpoint[i][0:12]
  print('More recent checkpoint is: ', checkfile_name)
  latest = checkpoint_dir + "/" + checkfile_name
else:
  last_cipher = 0
  load_checkpoints = False
  latest = 'NESSUN CHECKPOINT'
print('LATEST: ', latest)
checkpoint_path = root_dir + "CheckPoint/"+str(percentage)+"/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_path)
print(checkpoint_dir)
# Create a new model instance
# model = create_model()
MC = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, save_best_only=False, save_freq='epoch')


    
model = model_standard
opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, amsgrad=True)
model.compile(opt, loss = custom_loss, metrics=['mean_absolute_error', DSSIM, SSIM])

#vecchia versione
#ADM = Adam(lr=0.001, amsgrad=True)
#model.compile(optimizer = ADM, loss = custom_loss, metrics=['mean_absolute_error', SSIM])
    
# In[15]: controllare
if example:
  train_steps = 4 // batch_size
  val_steps = 4 // batch_size
else:
  train_steps = 3600 // batch_size
  val_steps = 900 // batch_size



# import keras
# print(keras.__version__)

# import tensorflow
# print(tensorflow.__version__)

# Load the previously saved weights
if load_checkpoints:
  model.load_weights(latest)
params = [percentage, LEARNING_RATE]
LogSaver = CustomLogSavingCallback(log_path=root_dir + "Results/results.json", parameters = params)

history = model.fit(train_generator, steps_per_epoch=train_steps,
                              validation_data=val_generator, validation_steps=val_steps,
                              epochs=EPOCHS, verbose=1, callbacks=[MC,TB, LogSaver])

print('HISTORY')
print(history)
print(dir(history))


# THIS IS WITHOUT CHECKPOINTS LOAD - ONLY FOR TESTING
#history = model.fit(train_generator, steps_per_epoch=train_steps,
#                              validation_data=val_generator, validation_steps=val_steps,
#                              epochs=3, verbose=1, callbacks=[ES, TB])


#model.save_weights(root_dir + "CheckPoint")

# Create a new model instance
#model = create_model()

# Restore the weights
#model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
#loss, acc = model.evaluate(test_images, test_labels, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

print(history.history.keys())
print(history.history['mean_absolute_error'])
print(history.history['val_mean_absolute_error'])
print(history.history['loss'])
print(history.history['val_loss'])
print(history.history['SSIM'])
print(history.history['val_SSIM'])



plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model mean_absolute_error')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['SSIM'])
plt.plot(history.history['val_SSIM'])
plt.title('model SSIM')
plt.ylabel('SSIM')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()