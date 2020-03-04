# -*- coding: UTF8 -*-

# cPickle是python2系列用的，3系列已經不用了，直接用pickle就好了
import pickle

# 重點是rb和r的區別，rb是開啟2進位制檔案，文字檔案用r
f=open('TechnicolorHijack.pkl','rb')
data=pickle.load(f)
print(data)