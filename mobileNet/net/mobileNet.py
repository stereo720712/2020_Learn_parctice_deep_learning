#coding:utf-8
'''
mobileNet
keypoint :
normal conv => Depthwise Conv + Pointwise
progress: =============================== wait train and run try
'''
import numpy as np
from keras.preprocessing import  image
from keras.models import Model

#DepthwiseConv2D
from keras.layers import  DepthwiseConv2D ,Input, Activation,Dropout,Reshape, BatchNormalization, \
    GlobalAveragePooling2D,Conv2D

from keras.applications.imagenet_utils import  decode_predictions
from keras import backend as K



def relu6(x):
    return K.relu(x, max_value=6)

#normal convolution block
def _conv_block(_inputs,_filters,_kernel=(3,3),_strides=(1,1),_block_id=1):

    x = Conv2D(_filters,_kernel,padding="same",use_bias=False,strides=_strides,name='conv1')(_inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6(x),name='conv1_relu')(x)


def _depthwise_conv_block(_inputs,_pointwise_conv_filters,_depth_multiplier=1,_strides=(1,1),_block_id=1):
    x = DepthwiseConv2D((3,3),padding='same',
                        deep_multiplier=_depth_multiplier,
                        strides=_strides,
                        use_bias=False,
                        name='conv_dw_%d' % _block_id)(_inputs)
    x = BatchNormalization(name='conv_pw_%d_bn' % _block_id)(x)
    x = Activation(relu6,name='conv_dw_%d_relu' & _block_id)(x)

    #pointwise convolution
    x = Conv2D(_pointwise_conv_filters,(1,1),padding='same',strides=(1,1),name='conv_pw_%d' % _block_id)(x)
    return Activation(relu6,name='conv_pw_%_relu' % _block_id)(x)




def MobileNetPra(input_shape=[224,224,3],
                depth_multiplier=1,
                dropout=1e-3,
                classes=1000):

    img_input = Input(shape=input_shape)

    #224,224,3 -> 112,112,64
    x = _conv_block(img_input,32,_strides=(2,2))

    #112,112,32 => 112,112,64
    x = _depthwise_conv_block(x,64,depth_multiplier,_block_id=1)

    #112,112,64 => 56,56,128
    x = _depthwise_conv_block(x,128,_strides=(2,2),_block_id=2)

    #56,56,128 = > 56,56,128
    x = _depthwise_conv_block(x,128,depth_multiplier,_block_id=3)

    #56,56,128 => 28,28,256
    x = _depthwise_conv_block(x,256,depth_multiplier,_strides=(2,2),_block_id=4)

    # 28,28,256 => 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, _strides=(2, 2), _block_id=5)

    #228,28,256 => 14,14,512
    x = _depthwise_conv_block(x,512,depth_multiplier,_strides=(2,2),_block_id=6)

    #============ 5 times dw and pw , 14,14,512=>14,14,512
    x = _depthwise_conv_block(x, 512,  depth_multiplier,_block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, _block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, _block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, _block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, _block_id=11)

    #14,14,512 => 7,7,1024
    x = _depthwise_conv_block(x,1024,depth_multiplier,_strides=(2,2),_block_id=12)
    x = _depthwise_conv_block(x,1024,depth_multiplier,_block_id=13)

    #7,7,1024  -> 1,1,1024
    x = GlobalAveragePooling2D(x)
    x = Reshape((1,1,1024),name='reshape1')(x)
    x = Dropout(dropout,name="dropout")(x)
    x = Conv2D(classes,(1,1),padding='same',name='conv_preds')(x)
    x = Activation('softmax',name='act_softmax')(x)

    input =img_input
    # check weight file path in some enviroment ??????????????????????????????????????????????????????????
    model = Model(input,x,name="mobilenet_1_0_224_tf")
    #check why not absoluate path
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)
    return model

#normal
def preprocess_input(x):
    x/255.
    x-=0.5
    x*=2
    return x

if __name__ == '__main__':
    model = MobileNetPra(input_shape=(224,224,3))
    img_path = 'elephant.jpg'
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    print('Input image shape: ',x.shape)

    # prediction
    preds = model.predict(x)
    print(np.argmax(preds))
    print('Prediected:',decode_predictions(preds,1))# display top 1






