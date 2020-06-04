# https://www.pyexercise.com/2019/01/rnn-lstm.html
# encoding:utf-8
 # =============================================================沒懂
#import IMDb example

from keras.datasets import imdb
from keras.preprocessing import sequence
# mac problem
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#  考慮最常見的一萬個字
max_features = 10000
# 最長句子
max_len = 200
batch_size = 32

print('Loaidng data...')

(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_features)

print(len(input_train),'train sqquence')
print(len(input_test),'test sequence')
# https://blog.csdn.net/Jiaach/article/details/79403352
print('Pad sequences (sample x time)')
input_train = sequence.pad_sequences(input_train,maxlen=max_len)
input_test = sequence.pad_sequences(input_test,maxlen=max_len)

print('input_train shape',input_train.shape)
print('input_test shape', input_test.shape)

# add RNN MODEL

from keras.layers import Dense,Embedding, SimpleRNN
from  keras.models import Sequential

#Simple RNN Define
model = Sequential()
#http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/
temp = Embedding(max_features,32)
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32,activation='tanh'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# set model 優化 損失函數 指標
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
summary = model.summary()
print(summary)
# train
history = model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

# #評估訓練模型的準確率 https://blog.csdn.net/Rex_WUST/article/details/84995954
acu = model.evaluate(input_test,y_test,verbose=1)
print(acu[1])


#draw train
# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1,len(acc)+1)
#
# plt.plot(epochs,acc, 'bo',label='Training acc')
# plt.plot(epochs,val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation accurancy')
# plt.legend()
# plt.figure()
#
# plt.plot(epochs,loss,'bo',label='Training loss')
# plt.plot(epochs,val_loss,'b',label='Vaildation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
