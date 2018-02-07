'''
This copy of code works as an example to train the neural networks with python interfaces	
		02/07/2018
		By Dong Nie
'''
import os
import h5py
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import skimage as ski
import scipy.ndimage as ndi
import random
import SimpleITK as sitk
import time

caffe.set_device(0) 
caffe.set_mode_gpu()

prefix_path = '/home/dongnie/caffe3/'
currprotoPath = prefix_path + 'path_to_your_train.prototxt'

outPath = '/home/dongnie/caffe3/'

isResume = False
resumeIter = 10000

maxIter = 1000000
base_trainIter = 1

showLossEvery = 2000
saveModelEvery = 5000
AdjustLossWeightEvery = 10000
AdjustLearningRateEvery = 20000

trainIter = 100
baselr = 0.0001 # from 0.001
lrgamma = 0.0001
thispower = 0.5


def ModifySolverPrototxt(currPath, trainIter, currIter, desiredLR, saveModelEvery):
    solver = caffe_pb2.SolverParameter()
    solver_file = prefix_path + '%_solver.prototxt'   # solver save path
    solver.train_net = currPath  # train file
    solver.test_net.append(currPath)  # test file
    solver.test_interval = 4000  # test interval
    solver.test_iter.append(100)  # test
    solver.max_iter = trainIter  # max iteration
    solver.base_lr = desiredLR  # base learning rate
    solver.momentum = 0.9  # momentum initial
    solver.weight_decay = 5e-4  # weight decay rate
    solver.lr_policy = 'inv'  # learning rate decay method
    solver.gamma = lrgamma  # learning rate decay rate
    solver.power = thispower
    solver.display = 100  # display interval
    solver.snapshot = saveModelEvery
    solver.snapshot_prefix = outPath + 'cross_train' + ''.format(currIter)
    solver.type = 'SGD'  # optimization method
    solver.solver_mode = caffe_pb2.SolverParameter.GPU
    with open(solver_file, 'w') as f:
        f.write(str(solver))



def getThisLr(currIter, lrgamma, baselr):
    desiredLR = baselr * (1 + lrgamma * currIter) ** (-thispower)

    return desiredLR

def main():
    #### Network Initialize #####
    currIter = 0
    maxIter = 100000
    net = caffe.Net('/path/prototxt/descriptor/file', caffe.TRAIN)
    
    ###Note, if you want to load the data manually:
    #net.blobs['data'].data = your_data
    #net.blobs['label'].data = your_label
    
    
    if isResume:
        startIter = resumeIter
    else:
		startIter = 1

    currIter += startIter
    ModifySolverPrototxt(currprotoPath, startIter, currIter, baselr, saveModelEvery)
    solver = caffe.SGDSolver('/path/to/solver/prototxt/file')
    #actually, you can set up the parameters yourself here.
        
    ##### begin training #########
    for i in range(startIter,maxIter):
        solver.step(1) #otherwise, you have to use solver.solve which optimize till the end
        
        if i%showLossEvery==0:
            netloss = [solver.net.blobs['lossX'].data,solver.net.blobs['lossY'].data,solver.net.blobs['lossF'].data]
            print 'Network Loss for Network  1, iteration  %d  is  Seg  %f, Cur  %f, Final  %f' % ( i, netloss[0], netloss[1], netloss[2])
        if i%saveModelEvery==0:
            
            print 'Saving cold started model for iteration: %d' % currIter
            solver.net.save(outPath + 'network_iter_%d.caffemodel' %(startIter))
            
        if i%AdjustLossWeightEvery==0:
            your_new_weight = 0
            net.params['loss_layer'].loss_weight[...] = your_new_weight
            
        if i%AdjustLearningRateEvery==0:
            desiredLR = 1e-4
            ModifySolverPrototxt(currprotoPath, startIter, currIter, desiredLR, saveModelEvery)
            solver = caffe.get_solver('temp.prototxt')

    
    
    ###### begin testing #######
    net.blobs['data'].data[...] = your_test_data
    net.forward()



if __name__ == '__main__':
    main()
