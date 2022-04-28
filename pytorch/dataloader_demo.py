from torch.utils.data import DataLoader
import numpy as np
import time
if __name__=="__main__":
    dataset=np.arange(1000)
    data=DataLoader(dataset=dataset,batch_size=100,  shuffle=False,num_workers=5,prefetch_factor=3)#
    begin=time.time()
    for i,d in enumerate(data):
        end=time.time()
        print('time:%f ms'%((end-begin)*1000))

        # if i>10:
        #     break
        begin=time.time()
    # datait=iter(data)
    # next(datait)