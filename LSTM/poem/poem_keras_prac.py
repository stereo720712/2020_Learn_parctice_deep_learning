'''
poem generate practice example

'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from  keras.layers import LSTM,Dense,Input,Softmax,Convolution1D,Embedding,Dropout
from  keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.models import Model
from poem_utils import load,get_batch,predict_from_nothing,predict_from_head


UNITS = 256
BATCH_SIZE = 64
EPOCHS = 50
POETRY_FILE = 'poetry.txt'

# load data

x_data, char2id_dict, id2char_dict = load(POETRY_FILE)
max_length = max([len(txt) for txt in x_data])
words_size = len(char2id_dict)

#========================#

# NN INIT

#=========================#

inputs = Input(shape=(None,words_size))
x = LSTM(UNITS,return_sequences=True)(inputs)
x =Dropout(0.4)(x)
x = LSTM(UNITS)(x)
x = Dropout(0.3)(x)
x = Dense(words_size,activation='softmax')(x)
model = Model(inputs,x)


# ----------------------------------------#
# split training set , validation set
# ----------------------------------------#

var_split = 0.1
np.random.seed(10101)
np.random.shuffle(x_data)
'''
，np.random.shuffle(X),只第一维度进行随机打乱，
体现在训练好像就是就是打乱每个样本的顺序，经常在epoch之后使用，
充分打乱数据这样可以增加训练的多样性
'''
np.random.seed(None)
num_val = int(len(x_data)*var_split)
num_train = len(x_data) - num_val

##-------------------------#
# set save data
#---------------------------#
checkpoint = ModelCheckpoint('logs/loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss',save_weights_only=True,save_best_only=False,period=1)

##https://blog.csdn.net/learning_tortosie/article/details/85243310
######################
# set learn and train
######################

model.compile(optimizer=Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])

for i in range(EPOCHS):
    predict_from_nothing(i,x_data,char2id_dict,id2char_dict,model)
    model.fit_generator(
        get_batch(BATCH_SIZE,x_data[:num_train],char2id_dict,id2char_dict),
        steps_per_epoch=max(1,num_train/BATCH_SIZE),
        validation_data=get_batch(BATCH_SIZE,x_data[:num_train],char2id_dict,id2char_dict),
        validation_steps=max(1,num_val/BATCH_SIZE),
        epochs=1,
        initial_epoch=0,
        callbacks=[checkpoint]
    )

######################
# set learn and train
######################

model.compile(optimizer=Adam(1e-4),loss='categorical_crossentropy',
              metrics=['accuracy'])
for i in range(EPOCHS):
   predict_from_nothing(i,x_data,char2id_dict,id2char_dict,model)
   model.fit_generator(
       get_batch(BATCH_SIZE,x_data[:num_train],char2id_dict,id2char_dict),
       steps_per_epoch=max(1,num_train/BATCH_SIZE),
       validation_data=get_batch(BATCH_SIZE,x_data[:num_train],char2id_dict,id2char_dict),
       validation_steps=max(1,num_val/BATCH_SIZE),
       epochs=1,
       initial_epoch=0,
       callbacks=[checkpoint]
   )
