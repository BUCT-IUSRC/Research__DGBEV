import pickle
# f = open('data/nuscenes/nuscenes_infos_val.pkl','rb')
f = open('/home/dell/wkq/BEVFusion-mit/run/full/official_mvp/test.pkl','rb')
data = pickle.load(f)
# f1 = open('/media/dell/hdd01/nuscenes/nuscenes/nuscenes_infos_val.pkl','rb')
# data1 = pickle.load(f1)
print(len(data))