import pickle
f = open('/data/nuscenes/nuscenes-full/nuscenes_infos_train.pkl','rb')   # 6019
data = pickle.load(f)
# # 从.pkl文件中加载数据
# with open(r'/data/nuscenes/nuscenes-full/nuscenes_infos_train.pkl', 'rb') as pkl_file:
#     data = pickle.load(pkl_file)

# 将数据写入.txt文件
with open('nuscenes_infos_train.txt', 'w') as txt_file:
    txt_file.write(str(data))
