#encoding:utf-8

import pandas as pd
import numpy as np
from sklearn import metrics

def get_relative_title():
    relative_title='''父、母、大姑、二姑、三姑、姑姑、二叔、三叔、大伯、二伯、姐姐、妹、第 、
    嫂 、姐夫、妹夫、大舅、二舅、老姨、二姨、大姨、岳父、女婿、外婆、小舅、
    媳妇、宝宝、宝贝儿、老婆 、亲爱、丈母娘、
    夫亲、郎君、夫君、良人、官人、相公、老公、爱人、卿卿、外子、外人、老头子、老伴、
    娘子、内人、良人、内子、老婆、爱人、卿卿、老婆子、老伴、
    爹、爹爹、爹亲、爹地、大大、老爸、爸比、爸、老爷子、
    娘娘、娘亲、娘妮、老妈、妈咪、妈、老娘、
    哥哥、兄长、兄台、兄亲、姊长、姊台、姊亲、
    弟、兄弟、弟子、弟亲、dad、mom、
    妹、姊妹、妹子、妹亲、妈、
    儿子、囝男、宝贝、孩子、女儿、闺女、囡女、丫头、宝贝、孩子、
    嫂子、兄嫂、兄妇、兄妻、老二、
    姐夫、姊兄、姊郎、姊丈、
    弟媳妇、兄弟媳妇、弟妹、弟妇、
    儿媳妇、半女、息妇、媳妇、三婶、大娘、二婶、
    爹、丈人、岳父、泰山、大大、老爸、爸、老爷子、
    娘、丈母娘、岳母、泰水、老妈、妈、老娘、
    爹、公公、公爹、大大、老爸、爸、老爷子、太上皇、
    娘、婆母娘、婆娘、老妈、妈、老娘、
    大舅哥、大舅子、小舅弟、小舅子、姑爷、老二、老三、
    妹、姊妹、妹子、小姨妹、小姨子、
    妹夫、妹弟、妹郎、小姑夫、表哥、大姐、大哥、二哥、三哥、三姐、二姐、大姐'''
    relative_title=set([ line.strip() for line in relative_title.split('、')  if len(line)>0])
    return relative_title


def is_relatives(word):
    relative_title=get_relative_title()
    spec_cond=('老婆' in word) or ('老公' in word) or ('母' in word) or ('宝贝' in word) or ('亲爱' in word) or (word =='姐') or (word =='哥')
    for rn in relative_title:
        if rn in word and len(word)<10:
            flag=1
            break
        elif  spec_cond:
            flag=1
            break
        else:
            flag=0
    return flag

def get_relative(pdSeris):
    relative_arr=np.array([is_relatives(name)  for name in pdSeris.values])
    return relative_arr


def ks(y_predicted, y_true):
    label=y_true
    #label = y_true.get_label()
    fpr,tpr,thres = metrics.roc_curve(label,y_predicted,pos_label=1)
    return 'ks',abs(fpr - tpr).max()


def MaxMin(df):
    return (df-df.min())/(df.max()-df.min())

def StanderScore(df):
    return (df-df.mean())/df.std()
