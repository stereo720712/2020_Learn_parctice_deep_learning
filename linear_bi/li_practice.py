#!
#coding:utf-8
# ref https://www.jianshu.com/p/ee6b1b51b43e
import cv2
import numpy as np
import matplotlib.pyplot as plt

def nearest_inter(ori_img,dst_height,dst_width):
    ori_height,ori_width,channel = ori_img.shape
    ratio_height = ori_height / dst_height
    ratio_width = ori_width / dst_width

    dst_img = np.zeros((dst_height,dst_width,channel),np.uint8)
    for y in range(dst_height):
        for x in range(dst_width):#location x , y
                row = ratio_height*y
                col = ratio_width*x
                near_row = round(row)
                near_col = round(col)
                if near_row == ori_height or near_col == ori_width:
                    near_row -= 1
                    near_col -= 1
                dst_img[y][x] = ori_img[near_row][near_col]
    return dst_img
'''
1.定位像素点

先找到目标图像像素点(dst_x, dst_y)在源图像上的像素点位置(src_x, src_y)。
一般是使用直接缩放：
src_x=dst_x * scale_x (scale_x为源图像与目标图像宽比例)
而我们这里使用几何中心对称：
src_x = (dst_x + 0.5) * scale_x - 0.5
然后找到上下左右最近邻的四个像素点用于计算插值。

ref : https://blog.csdn.net/wudi_X/article/details/79782832

2.two step interpolation
设(x,y)为插值点在源图像的坐标，待计算插值为z
首先x方向上插值：相邻两点为(x0,y0)、(x1,y0)，像素值分别为
z0 = f(x0,y0) , z1 = f(x1,y0)
由單線性推得 z-z0/x-x0 = z1-z0/x1-x0
==> 可推得 z = (x1-x)/(x1-x0) * z0 + (x-x0)/(x1-x0) * z1 
然后y方向上插值：以上得到的上方插值z记为ztop，同理可得下方插值为zbot，那么最后插值为：
Z = (y1-y)/(y1-y0) * Ztop + (y-y0)/(y1-y0) * Zbot

'''
def bilinear(src_img,dst_hight,dst_width):
    o_height,o_width,o_channel = src_img.shape
    bi_img = np.zeros(shape=(bigger_height,bigger_width,o_channel),dtype=np.uint8)
    ratio_height = float(o_height) / dst_hight
    ratio_width = float(o_width) / dst_width
    for n in range(3):
        for dst_y in range(dst_hight):
            for dst_x in range(dst_width):
                #target point in source img
                src_x = (dst_x + 0.5) * ratio_width - 0.5
                src_y = (dst_y + 0.5) * ratio_height - 0.5

                # calculate the target neighbor four points surround target
                src_x_0 = int(np.floor(src_x))
                src_y_0 = int(np.floor(src_y))
                src_x_1 = min(src_x_0 + 1,o_width -1) #max = original width
                src_y_1 = min(src_y_0 + 1,o_height -1) # max = original height

                #bilinear interpolation
                value0 = (src_x_1 - src_x) * src_img[src_y_0,src_x_0,n] + (src_x - src_x_0) * src_img[src_y_0,src_x_1,n]
                value1 = (src_x_1 - src_x) * src_img[src_y_1,src_x_0,n] + (src_x - src_x_0) * src_img[src_y_1,src_x_1,n]
                bi_img[dst_y,dst_x,n] = int((src_y_1 - src_y) *value0 + (src_y - src_y_0) * value1)

    return bi_img




if __name__ == '__main__':
  
  img = cv2.imread('lenna.jpg',cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  bigger_height = img.shape[0] + 300
  bigger_width = img.shape[1] + 300
  near_img = nearest_inter(img,bigger_height,bigger_width)
  bilinear_img = bilinear(img,bigger_height,bigger_width)

  plt.figure()
  plt.subplot(2,2,1)
  plt.title("Nearest Image")
  plt.imshow(near_img)
  plt.subplot(2,2,2)
  plt.title("Bilinear Image")
  plt.imshow(bilinear_img)
  plt.subplot(2,2,3)
  plt.title("Source Image")
  plt.imshow(img)
  plt.show()




if __name__ == '__main__':
  
  img = cv2.imread('lenna.jpg',cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  bigger_height = img.shape[0] + 200
  bigger_width = img.shape[1] + 200
  near_img = nearest_inter(img,bigger_height,bigger_width)

  plt.figure()
  plt.subplot(2,2,1)
  plt.title("Nearest Image")
  plt.imshow(near_img)
  plt.show()

