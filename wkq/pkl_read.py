# cPickle是python2系列用的，3系列已经不用了，直接用pickle就好了
import pickle
# f = open('data/nuscenes/nuscenes_infos_val.pkl','rb')
f = open('/data/nuscenes/nuscenes-full/nuscenes_infos_test.pkl','rb')
# f = open('/home/dell/wkq/BEVFusion-mit/run/full/official_mvp/test.pkl','rb')
data = pickle.load(f)

# print((data.keys()))
# print(len(data['infos']))
# print(type(data['infos']))
# # print((data['metadata']["version"]))
# print((data['infos'][0].keys()))

# print(type(data))
# print(len(data))
# print(data[0].keys())
# print(type(data[0]))



# data1={}
# data1['infos']=data['infos'][0:80]
# data1['metadata']=data['metadata']
# shoplistfile = 'data/nuscenes/new_0_80.pkl'  #保存文件数据所在文件的文件名
# print(len(data1['infos']))
# f = open(shoplistfile, 'wb') #二进制打开，如果找不到该文件，则创建一个
# pickle.dump(data1, f) #写入文件

data1={}
data1['infos']=data
data1['metadata']=data['metadata']
shoplistfile = 'data/nuscenes/new_test.pkl'  #保存文件数据所在文件的文件名
print(len(data1['infos']))
f = open(shoplistfile, 'wb') #二进制打开，如果找不到该文件，则创建一个
pickle.dump(data1, f) #写入文件

f.close()  #关闭文件
del data # 删除列表

