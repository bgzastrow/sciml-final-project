import matplotlib.pyplot as plt
import numpy as np      
import sys
import torch
from time import time
import datetime             as dt
from tqdm       import tqdm

import src.mace.mace            as mace
from src.mace.loss              import Loss
import src.mace.loss            as loss  
import src.mace.utils           as utils
import src.mace.train as train
import src.mace.load            as load

start = time()
now = dt.datetime.now()
name = str(now.strftime("%Y%m%d")+'_'+now.strftime("%H%M%S"))


arg_type = sys.argv[1]
arg_file = sys.argv[2]
dirname = sys.argv[3]
infile = './input/'+arg_type+'/'+arg_file+'.in'
path = './models/'+arg_type+'/'+dirname
outloc = './models/'+arg_type+'/'

utils.makeOutputDir(path+'/test')

## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1 
kwargs = {'num_workers': 1, 'pin_memory': True} 

if arg_type != 'CSE' and arg_type != 'L96':
    raise ValueError('Model type ' + arg_type + ' unknown. Select either CSE or L96')

## CHOOSE DATASET -- CSE OR L96
if arg_type == 'CSE':
    from src.mace.CSE.input import Input
    import src.mace.CSE.test as test
    input_data = Input(infile, name)
    import src.mace.CSE.dataset as ds
    traindata, testdata, data_loader, test_loader = ds.get_data(dt_fract=input_data.dt_fract,
                                                                nb_samples=input_data.nb_samples, batch_size=batch_size, 
                                                                nb_test=input_data.nb_test,kwargs=kwargs)
    trained = load.Trained_MACE(p_dim=4, outloc=outloc, dirname=dirname, epoch=5)
elif arg_type == 'L96':
    from src.mace.L96.input import Input
    import src.mace.L96.test as test
    input_data = Input(infile, name)
    import src.mace.L96.dataset as ds
    traindata, testdata, data_loader, test_loader = ds.get_data(dt_fract=input_data.dt_fract, n_L96=input_data.n_dim,
                                                                nb_samples=input_data.nb_samples, batch_size=batch_size, 
                                                                nb_test=input_data.nb_test,kwargs=kwargs)
    trained = load.Trained_MACE(p_dim=1, n_dim=input_data.n_dim, outloc=outloc, dirname=dirname, epoch=5)


meta = trained.get_meta()
model = trained.model
## Test the model on the test samples

print('\n\n>>> Testing model on test samples ...')

sum_err_step = 0
sum_err_evol = 0

step_calctime = list()
evol_calctime = list()
if arg_type == 'CSE':
    for i in tqdm(range(len(traindata.testpath))):
    #     print(i+1,end='\r')
        testpath = traindata.testpath[i]
        err_test, err_evol, step_time, evol_time,n, n_hat, n_evol  = test.test_model(model, testpath, meta, plotting=True, save=True)

        sum_err_step += err_test
        sum_err_evol += err_evol

        step_calctime.append(step_time)
        evol_calctime.append(evol_time)
elif arg_type == 'L96':
    for i in tqdm(range(len(traindata.testF))):
        testF = traindata.testF[i] 
        err_test, err_evol, step_time, evol_time,n, n_hat, n_evol  = test.test_model(model, input_data.n_dim, testF, meta, plotting = True, save=True)
        sum_err_step += err_test
        sum_err_evol += err_evol

        step_calctime.append(step_time)
        evol_calctime.append(evol_time)

utils.makeOutputDir(path+'/test')

if arg_type == 'CSE':
    np.save(path+ '/test/sum_err_step.npy', np.array(sum_err_step/len(traindata.testpath)))
    np.save(path+ '/test/sum_err_evol.npy', np.array(sum_err_evol/len(traindata.testpath)))
elif arg_type == 'L96':
    np.save(path+ '/test/sum_err_step.npy', np.array(sum_err_step/len(traindata.testF)))
    np.save(path+ '/test/sum_err_evol.npy', np.array(sum_err_evol/len(traindata.testF)))

np.save(path+ '/test/calctime_evol.npy', evol_calctime)
np.save(path+ '/test/calctime_step.npy', step_calctime)  

print('\nAverage error:')
print('           Step:', np.round(sum_err_step,3))
print('      Evolution:', np.round(sum_err_evol,3))
print('(following Eq. 23 of Maes et al., 2024)')

stop = time()

print('\n>>> FULLY DONE!')

total_time = stop-start
if total_time < 60:
        print('Total time [secs]:', np.round(total_time,2))
if total_time >= 60:
        print('Total time [mins]:', np.round(total_time/60,2))
if total_time >= 3600:
        print('Total time [hours]:', np.round(total_time/3600,2))

print('Output saved in:', path,'\n') 
