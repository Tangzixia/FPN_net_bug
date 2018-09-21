#coding=utf-8
#参考：https://blog.csdn.net/u013040887/article/details/81048689
from models.fpn_net import *
import tensorflow as tf
import argparse
import sys
import os
import numpy as np
from PIL import Image

flag=tf.flags
flag.DEFINE_string("hello",None,"description")
flag.DEFINE_integer("max_iter",20000,"description")
flag.DEFINE_string("logdir","./training","")
FLAG=flag.FLAGS
def get_data_and_label(fileName, filePath=CAPTCHA_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    #转为灰度图
    img = img.convert("L")
    image_array = np.array(img)
    image_data = image_array.flatten()/255
    image_label = name2label(fileName[0:CAPTCHA_LEN])
    return image_data, image_label
def gen_next_batch(batch_size=32,mode="TRAIN",step=0):
    batch_data = np.zeros([batchSize, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT])
    batch_label = np.zeros([batchSize, CAPTCHA_LEN * CHAR_SET_LEN])
    fileList=trainList
    if mode=="VALID":
        fileList=valList

    totalNumber=len(fileList)
    indexStart=step*batch_size
    for i in range(batch_size):
        index=(i+indexStart)%totalNumber
        name = fileList[index]
        img_data,img_label=get_data_and_label(name)
        batch_data[i,:]=img_data
        batch_label[i,:]=img_label
    return batch_data,batch_label
def parse_args(args):
    parser=argparse.ArgumentParser()
    parser.add_argument("--iters",dest="max_iter",default="hello",type=str)
    parser.add_argument("--weights",dest="pretrained_weights",default=1,type=int)
    parser.add_arguments("--logdir",dest="logdir",default="",type=str)
    return parser.parse_args(args)
def main(args=None):
    if args is not None:
        args=sys.argv[1:]
    else:
        args=parse_args(args)

    x=tf.placeholder([None,None,None,3],dtype=tf.int32)
    y=tf.placeholder([None,1000],dtype=tf.int32)

    model=FPN()
    logits,endpoints,varlist=model.build_model(x)
    logits_s=tf.nn.softmax(logits)

    losses=model.losses(logits,y)

    correct_pred = tf.equal(tf.argmax(logits_s, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar("losses",losses)

    merged_summary=tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
    optimilizer=tf.train.AdamOptimizer(learning_rate=0.01,beta=0.09)
    train_step=optimilizer.minimize(losses,varlist)

    saver=tf.train.Saver(var_list=varlist)
    init=tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer=tf.summary.FileWriter(args.logdir,sess.graph)
        for i in range(args.max_iter):
            batch_images,batch_labels=get_next_batch()
            train_dict={x:batch_images,y:batch_labels}

            sess.run(train_step,feed_dict=train_dict)
            summary,losses,accuracy=sess.run([merged_summary,losses,acc],feed_dict=train_dict)
            print("第{}次训练的loss:{},准确率: {}".format(i,losses,accuracy))
            writer.add_summary(summary,i+1)
            if((i+1)%100==0):
                val_images,val_labels=get_next_batch(64,"VALID")
                feed_dict={x:val_images,y:val_labels}
                val_acc=sess.run(acc,feed_dict=feed_dict)
                print("第{}次验证集上的acc: {1:.2f}".format(i,acc))
                if(val_acc>0.99):
                    model_save_path = os.path.join(args.logdir, "model.ckpt")
                    saver.save(sess, model_save_path, global_step=i + 1)
                    break
            if((i+1)%1000==0):
                model_save_path=os.path.join(args.logdir,"model.ckpt")
                saver.save(sess,model_save_path,global_step=i+1)
        writer.close()


if __name__=="__main__":
    main()
