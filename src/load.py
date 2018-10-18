import os
import numpy as np

def load(dir_='../data/npy'):
    x_train=np.load(os.path.join(dir_,'x_train.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test.npy'))
    x_test_1 = np.load(os.path.join(dir_, 'x_test_1.npy'))
    x_test_2 = np.load(os.path.join(dir_, 'x_test_2.npy'))

    return x_train,x_test,x_test_1,x_test_2
if __name__=='__main__':
    x_train,x_test,x_test_1,x_test_2=load()
    print (x_train.shape)
    print (x_test.shape)
    print(x_test_1.shape)
    print(x_test_2.shape)

