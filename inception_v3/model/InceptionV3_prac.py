# ===========================================
# Inception V3 Model Practice
# ===========================================



import numpy as np

from keras.models import Model
from keras import  layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D #??!?
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


# convolution with bn
def conv2d_bn(x,filters,kernel_row,kernel_col,strides=(1,1), padding='same',name="None"):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(filters,
               (kernel_row,kernel_col),
               strides=strides,
               padding=padding,
               use_bias=False,
               name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu',name=name)(x)
    return  x

# ref https://blog.csdn.net/weixin_39881922/article/details/80346070
# Inception_V3 Model Define

def InceptionV3_prac(input_shape=[299,299,3],classes=1000):

    img_input = Input(shape=input_shape)
    x = conv2d_bn(img_input,32,3,3,strides=(2,2),padding='valid')
    x = conv2d_bn(x,32,3,3,padding='valid')
    x = conv2d_bn(x,64,3,3)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv2d_bn(x,80,1,1,padding='valid')
    x = conv2d_bn(x,192,3,3,padding='valid')
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    # #===============================##
    # #Inception Block1 35x35
    # #===============================##

    # Block1 part1
    # 35 x35 x 192 => 35 x 35 x 256

    branch1x1 = conv2d_bn(x, 64, 1, 1) # 35,35,64

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64,1,1) # 35x35x32
    # 跟 ref 看到的 kernel size 不同
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5) # 35x35x64

    branch3x3 = conv2d_bn(x,64,1,1)
    branch3x3 = conv2d_bn(branch3x3,96,3,3)
    branch3x3 = conv2d_bn(branch3x3,96,3,3) #35,35,96

    # 64 + 64 +96 + 32
    x = layers.concatenate([branch1x1,branch5x5,branch3x3,branch_pool],axis=3,
                           name='mixed1-0') #not very clear

    # Block1 part2
    # 35 x35 x 256 => 35 x 35 x 288
    branch1x1 = conv2d_bn(x,64,1,1)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='smae')(x)
    branch_pool = conv2d_bn(branch_pool,64,1,1)

    branch5x5 = conv2d_bn(x,48,1,1)
    branch5x5 = conv2d_bn(branch5x5,64,5,5)

    branch3x3 = conv2d_bn(x,64,1,1)
    branch3x3 = conv2d_bn(branch3x3,96,3,3)
    branch3x3 = conv2d_bn(branch3x3,96,3,3)

    # 64 + 64 + 96 + 64 ,288
    x = layers.concatenate([branch1x1,branch5x5,branch3x3,branch_pool],axis=3,
                           name='mix1-1')

    # Block1 part3
    # 35 x 35 x 288 => 35 x 35 x 288
    branch1x1 = conv2d_bn(x,64,1,1)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='smae')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3 = conv2d_bn(x, 64, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)
    branch3x3 = conv2d_bn(branch3x3, 96, 3, 3)

    x = layers.concatenate([branch1x1,branch_pool,branch3x3,branch5x5],axis=3,
                           name="mix1-2")

    # =======================#
    # Block2 17 x 17
    # =======================#

    # Block2 part1
    # 35 x35x 288 => 17 x 17 x 768

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2,2),padding='valid')

    branch3x3dl = conv2d_bn(x, 64, 1, 1)
    branch3x3dl = conv2d_bn(branch3x3dl, 96, 3, 3)
    branch3x3dl = conv2d_bn(branch3x3dl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)
    x = layers.concatenate([branch3x3,branch3x3dl,branch_pool],axis=3,name='mix2-1')

    # Block part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed2-2')

    # Block2 part3 and part4
    # 17 x 17 x 768 => 17 x 17 x 768 => 17 x 17 x 768
    for i in range(2):
        branch1x1 =conv2d_bn(x,192, 1,1)

        branch7x7 = conv2d_bn(x,160,1,1)
        branch7x7 = conv2d_bn(branch7x7,160,1,7)
        branch7x7 = conv2d_bn(branch7x7,160,7,1)

        branch7x7dbl = conv2d_bn(x,160,1,1)
        branch7x7dbl = conv2d_bn(branch7x7dbl,160,7,1)
        branch7x7dbl = conv2d_bn(branch7x7dbl,160,1,7)
        branch7x7dbl = conv2d_bn(branch7x7dbl,160,7,1)
        branch7x7dbl = conv2d_bn(branch7x7dbl,160,7,1)

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2d_bn(branch_pool,192,1,1)

        x = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name='mix-3-3&4')


     # Block2 part5
     # 17 x 17 x 768 => 17 x 17 x 768

    branch1x1 = conv2d_bn(x,192,1,1)

    branch7x7 = conv2d_bn(x,192,1,1)
    branch7x7 = conv2d_bn(branch7x7,7,1)
    branch7x7 = conv2d_bn(branch7x7,1,7)

    branch7x7dbl = conv2d_bn(x,192, 1,1)
    branch7x7dbl = conv2d_bn(branch7x7dbl,192,7,1)
    branch7x7dbl = conv2d_bn(branch7x7dbl,192,1,7)
    branch7x7dbl = conv2d_bn(branch7x7dbl,192,7,1)
    branch7x7dbl = conv2d_bn(branch7x7dbl,192,1,7)

    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool =conv2d_bn(branch_pool,192,1,1)
    x = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],axis=3,name="mix3=5")


    # =======================#
    # Block3 8 x 8
    # =======================#

    # Block3 part1
    # 17 x 17 x 768 => 8 x 8 x 1280
    branch3x3 = conv2d_bn(x,192,1,1)
    branch3x3 = conv2d_bn(branch3x3,320,3,3,strides=(2,2),padding='valid')

    branch7x7x3 = conv2d_bn(x,192,1,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,1,7)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,7,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,3,3,strides=(2,2),padding='valid')

    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)

    x= layers.concatenate([branch3x3,branch7x7x3,branch_pool],axis=3,name='mix3-1')

    # Block3 part2 & 3
    # 8 x 8 x 1280 => 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x,320,1,1)

        branch3x3 = conv2d_bn(x,384,1,1)
        branch3x3_1 = conv2d_bn(branch3x3,384,1,3)
        branch3x3_2 = conv2d_bn(branch3x3,384,3,1)
        branch3x3 = layers.concatenate([branch3x3_1,branch3x3_2],axis=3,name="mix3-2" + str(i))

        branch3x3dl = conv2d_bn(x,448,1,1)
        branch3x3dl = conv2d_bn(branch3x3dl,384,1,3)
        branch3x3dl_1 = conv2d_bn(branch3x3dl,1,3)
        branch3x3dl_2 = conv2d_bn(branch3x3dl,3,1)
        branch3x3dl = layers.concatenate([branch3x3dl_1,branch3x3_2],axis=3)

        branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
        branch_pool = conv2d_bn(branch_pool,192,1,1)
        x = layers.concatenate([branch1x1,branch3x3,branch3x3dl,branch_pool],axis=3,name="mixed"+str(9+i))

    #pooling and connect
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes,activation='softmax',name="predict")(x)

    inputs = img_input

    model = Model(inputs,x,name='inception_V3')

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = InceptionV3_prac()
    model.laod_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    img_path = "'elephant.jpg"
    img = image.load_img(img_path,target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Prediected: ',decode_predictions(preds) )
    




