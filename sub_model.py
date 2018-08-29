import cPickle
import numpy as np
import os

def unpickle(fid):
    with open(fid, 'rb') as fo:
        data = cPickle.load(fo)
    return data

def pickle(fid,data):
    with open(fid, 'wb') as fo:
        cPickle.dump(data, fo)

def generate_data(mode, TARGET_ODER):
    fid = "/data/srd/data/Image/cifar-100-python/" + mode
    data = unpickle(fid)

    feats = data["data"]
    labs = data["fine_labels"]
    
    file_path = "/data/srd/data/Image/cifar-100-"+str(TARGET_ODER)

    cfm = np.fromfile("/data/srd/data/Image/cifar-100-python/cmf.bin", dtype=np.int64)
    cfm = cfm.reshape(100,100)
    TOP = 3
    new_dic_list = cfm[:,TARGET_ODER].argsort()[-TOP:][::-1]

    if not os.path.exists(file_path):
        os.mkdir("/data/srd/data/Image/cifar-100-"+str(TARGET_ODER))
    
    dic_5_name = file_path+'/dic_3.bin'
    new_dic_list.tofile(dic_5_name)

    n = len(labs)
    feats_3 = []
    labs_3 = []
    for i in range(n):
        if labs[i] in new_dic_list:
            feats_3.append(feats[i])
            labs_3.append(np.where(new_dic_list==labs[i])[0][0])

    feats_3 = np.array(feats_3)

    data["data"] = feats_3
    data["fine_labels"] = labs_3

    pickle(file_path+"/"+mode,data)

for i in range(100):
    generate_data("train", i)
    generate_data("test", i)

    os.system("python train_densenet.py --depth 100 --growth_rate 12 --model_name densenet-"+str(i)+" --pre_train densenet4cifar100_2 --lr 0.001 --epoch 200 --fid /data/srd/data/Image/cifar-100-"+str(i)+"/ --nclass 3 --finetune 1")
    os.system("python train_densenet.py --depth 100 --growth_rate 12 --pre_train densenet-"+str(i)+" --fid /data/srd/data/Image/cifar-100-python/ --nclass 3 --onlyevalue 1")


