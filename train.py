# -*- coding: utf-8 -*-

from model import Deeplabv3,get_unet
import numpy as np
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)


#可以自定义
N_Cls = 6
ISZ = 160
smooth = 1e-12
dirs="data/"

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jac_mean=K.mean(jac)
    return jac_mean

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))#clip剪切
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2]) 
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def calc_jacc(model,xval,yval):
    img = xval
    msk = yval

    prd = model.predict(img, batch_size=4)
    print('shape: ',prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(N_Cls):
        t_msk = msk[:, :, :,i]
        t_prd = prd[:, :, :,i]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[1], msk.shape[2])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[1], msk.shape[2])

        m, b_tr = 0, 0
        for j in range(10):
            tr = j/10.0
            pred_binary_mask = t_prd > tr
            jk = jaccard_npcoef(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print(i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 6.0
    return score, trs

def train_net(img,msk,x_val,y_val):    
    print("start train net")
    s=0
    model = Deeplabv3(input_shape=(160,160,8), classes=6,backbone='xception')
    #model.load_weights('weights/deeplab_x_jk0.6541', by_name=True)
    #model = Deeplabv3(input_shape=(160,160,8), classes=6,backbone='mobilenetv2')
    #model.load_weights('weights/deeplab_m_jk0.6479', by_name=True)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    #model_checkpoint = ModelCheckpoint('weights/deeplab_tmp.h5', monitor='loss', save_best_only=True)
    for i in range(6):
        model.fit(img, msk, batch_size=4, epochs=1, verbose=1, shuffle=True, validation_data=(x_val, y_val))
        #model.fit(img, msk, batch_size=8, epochs=1, verbose=1, shuffle=True, validation_split=0.2)
        score, trs = calc_jacc(model,x_val,y_val)
        print('val jk', score)
        if score >s:
            model.save_weights('weights/deeplab_x_jk%.4f' % score)
            s=score
    return model,trs

#numpy计算jaccard
def jaccard_npcoef(y_true, y_pred):
    intersection = np.sum(np.sum(y_true * y_pred, axis=0),axis=0)
    sum_ = np.sum(np.sum(y_true + y_pred, axis=0),axis=0)
    #print('sum.shape: ',sum_.shape,intersection.shape)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jac_mean=np.mean(jac)
    return jac_mean
    
#训练unet
def train_unet(img,msk,x_val,y_val):
    print ("start train net")
    s=0
    model = get_unet()    
    #model.load_weights('weights/unet_jk0.6198',by_name=True)
    #model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)  #save temp model
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    for i in range(5):
        model.fit(img, msk, batch_size=32, epochs=1, verbose=1, shuffle=True, validation_data=(x_val, y_val))
        score,trs= calc_jacc(model,x_val,y_val)
        print ('val jk', score)
        if score >s:
            model.save_weights('weights/unet_jk%.4f' % score)
            s=score
    return model,trs
    
def predic(model,xval,yval):
    img = xval
    msk = yval

    prd = model.predict(img, batch_size=4)
    print('shape: ',prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(N_Cls):
        t_msk = msk[:, :, :,i]
        t_prd = prd[:, :, :,i]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[1], msk.shape[2])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[1], msk.shape[2])        
        pred_binary_mask = np.round(t_prd)
        jk = jaccard_npcoef(t_msk, pred_binary_mask)        
        print(i, jk)
        avg.append(jk)
    score = sum(avg) / 6.0
    return score

if __name__ == '__main__':
    #calc_trs()

    x_val = np.load(dirs+'x161_eval.npy')
    y_val = np.load(dirs+'y161_eval.npy')
    img = np.load('data/x161_train.npy')#shape-(4175*4175*8)
    msk = np.load('data/y161_train.npy')

    sample_indices = np.random.choice(img.shape[0], size=1000, replace=False)
    img_reduced = img[sample_indices]
    msk_reduced = msk[sample_indices]

    x_val_reduced = x_val[:200]
    y_val_reduced = y_val[:200]

    # img_reduced = tf.image.resize(img_reduced, (128, 128))
    # msk_reduced = tf.image.resize(msk_reduced, (128, 128))
    # x_val_reduced = tf.image.resize(x_val_reduced, (128, 128))
    # y_val_reduced = tf.image.resize(y_val_reduced, (128, 128))

    # img = img.astype('float32')
    # msk = msk.astype('float32')
    # x_val = x_val.astype('float32')
    # y_val = y_val.astype('float32')
    #model ,x1= train_net(img,msk,x_val,y_val)
    #model ,x1= train_unet(img,msk,x_val,y_val)
    #model, x1 = train_unet(img_reduced, msk_reduced, x_val_reduced, y_val_reduced)
    model ,x1= train_net(img_reduced, msk_reduced, x_val_reduced, y_val_reduced)
    #calc_jaccard(model,x1)
    #prd = model.predict(img[1:2,:,:,:])



    '''
    #计算模型消耗时间
    start=time.time()
    prd = model.predict(img, batch_size=4)
    end=time.time()
    tt=end-start
    '''
