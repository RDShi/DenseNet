import os
import time

start=time.time()

depth = 100
growth_rate = 12

lr = 0.1
epoch = 150
model_name = "densenet4cifar100_0"
os.system("python train_densenet.py --model_name "+model_name+" --lr "+str(lr)+" --epoch "+str(epoch)+ 
          " --depth "+str(depth)+" --growth_rate "+str(growth_rate))

lr = 0.01
epoch = 75
model_name = "densenet4cifar100_1"
pre_train = "densenet4cifar100_0"
os.system("python train_densenet.py --model_name "+model_name+" --lr "+str(lr)+" --epoch "+str(epoch)+" --pre_train "+str(pre_train)+
          " --depth "+str(depth)+" --growth_rate "+str(growth_rate))


lr = 0.001
epoch = 75
model_name = "densenet4cifar100_2"
pre_train = "densenet4cifar100_1"
os.system("python train_densenet.py --model_name "+model_name+" --lr "+str(lr)+" --epoch "+str(epoch)+" --pre_train "+str(pre_train)+
          " --depth "+str(depth)+" --growth_rate "+str(growth_rate))

print("consuming time",time.time()-start)
