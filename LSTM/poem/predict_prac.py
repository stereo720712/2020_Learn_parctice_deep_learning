
# encoding:utf-8
import  os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from poem_utils import  load,get_batch,predict_from_nothing,predict_from_head
from keras.models import Model
from keras.layers import  LSTM,Dense,Input,Softmax,Convolution1D,Embedding,Dropout

UNITS = 256
POERTY_FILE = './poetry.txt'

#load data
x_data ,char2id_dict,id2char_dict = load(POERTY_FILE)
max_length = max([len(txt) for txt in x_data])
words_size = len(char2id_dict)

##-------------------------
## Declare Model
##-------------------------

inputs = Input(shape=(None,words_size))
x = LSTM(UNITS,return_sequences=True)(inputs)
x = Dropout(0.3)(x)
x = LSTM(UNITS)(x)
x = Dropout(0.3)(x)
x = Dense(words_size,activation='softmax')(x)
model = Model(inputs,x)

model.load_weights("logs/loss4.419-val_loss4.009.h5")
#predict_from_nothing(0,x_data,char2id_dict,id2char_dict,model)
predict_from_head("鹏飞牛逼",x_data,char2id_dict,id2char_dict,model)
predict_from_head("飞哥好帅",x_data,char2id_dict,id2char_dict,model)
predict_from_head("妹爱飞哥",x_data,char2id_dict,id2char_dict,model)
predict_from_head("飞哥屌大",x_data,char2id_dict,id2char_dict,model)
predict_from_head("鹏飞帅呆",x_data,char2id_dict,id2char_dict,model)