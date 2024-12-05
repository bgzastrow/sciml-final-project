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

specs_dict, idx_specs = utils.get_specs()

start = time()
now = dt.datetime.now()
name = str(now.strftime("%Y%m%d")+'_'+now.strftime("%H%M%S"))

## ================================================== INPUT ========
## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS

## READ INPUT FILE
arg_type = sys.argv[1]
arg_file = sys.argv[2]
infile = './input/'+arg_type+'/'+arg_file+'.in'
path = './models/'+arg_type+'/'+name

## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1 
kwargs = {'num_workers': 1, 'pin_memory': True} 

if arg_type != 'CSE' and arg_type != 'L96':
    raise ValueError('Model type ' + arg_type + ' unknown. Select either CSE or L96')

utils.makeOutputDir(path)
utils.makeOutputDir(path+'/nn')

## CHOOSE DATASET -- CSE OR L96
if arg_type == 'CSE':
    import src.mace.CSE.train as train
    from src.mace.CSE.input import Input
    import src.mace.CSE.test as test
    input_data = Input(infile, name)
    input_data.print()
    import src.mace.CSE.dataset as ds
    traindata, testdata, data_loader, test_loader = ds.get_data(dt_fract=input_data.dt_fract,
                                                                nb_samples=input_data.nb_samples, batch_size=batch_size, 
                                                                nb_test=input_data.nb_test,kwargs=kwargs)
elif arg_type == 'L96':
    import src.mace.L96.train as train
    from src.mace.L96.input import Input
    import src.mace.L96.test as test
    import src.mace.L96.dataset as ds
    N_L96 = 100 # dimension of L96
    traindata, testdata, data_loader, test_loader = ds.get_data(dt_fract=input_data.dt_fract, n_L96=N,
                                                                nb_samples=input_data.nb_samples, batch_size=batch_size, 
                                                                nb_test=input_data.nb_test,kwargs=kwargs)

meta = input_data.make_meta(path)
## Make model
model = mace.Solver(n_dim=input_data.n_dim, p_dim=4,z_dim = input_data.z_dim, 
                    nb_hidden=input_data.nb_hidden, ae_type=input_data.ae_type, 
                    scheme=input_data.scheme, nb_evol=input_data.nb_evol,
                    path = path,
                    DEVICE = DEVICE,
                    lr=input_data.lr )

num_params = utils.count_parameters(model)
print(f'\nThe model has {num_params} trainable parameters')

## ================================================== TRAIN ========

## ------------- PART 1: unnormalised losses ----------------
norm, fract = loss.initialise()

## Make loss objects
trainloss = Loss(norm, fract, input_data.losstype)
testloss  = Loss(norm, fract, input_data.losstype)

## Train
tic = time()
train.train(model, 
            data_loader, test_loader, 
            end_epochs = input_data.ini_epochs, 
            trainloss=trainloss, testloss=testloss, 
            start_time = start)
toc = time()
train_time1 = toc-tic


## ------------- PART 2: normalised losses, but reinitialise model ----------------

## Change the ratio of losses via the fraction
print('\n\n>>> Continue with normalised losses.')

fract = input_data.get_facts()
trainloss.change_fract(fract)
testloss.change_fract(fract)

## Normalise the losses
new_norm = trainloss.normalise()  
testloss.change_norm(new_norm) 

## Continue training
tic = time()
train.train(model, 
            data_loader, test_loader, 
            start_epochs = input_data.ini_epochs, end_epochs = input_data.nb_epochs, 
            trainloss=trainloss, testloss=testloss, 
            start_time = start)
toc = time()
train_time2 = toc-tic

train_time = train_time1 + train_time2


## ================================================== SAVE ========


## losses
trainloss.save(path+'/train')
testloss.save(path+'/valid')

## dataset characteristics
min_max = np.stack((traindata.mins, traindata.maxs), axis=1)
np.save(path+'/minmax', min_max) 

## model
torch.save(model.state_dict(),path+'/nn/nn.pt')

## status
np.save(path+'/train/status', model.get_status('train')) # type: ignore
np.save(path +'/valid/status', model.get_status('test') ) # type: ignore

fig_loss = loss.plot(trainloss, testloss, len = input_data.nb_epochs)
plt.savefig(path+'/loss.png')

stop = time()

overhead_time = (stop-start)-train_time

## updating meta file
input_data.update_meta(traindata, train_time, overhead_time, path)

## ================================================== TEST ========

input_data.print()

## Test the model on the test samples

print('\n\n>>> Testing model on',len(traindata.testpath),'test samples ...')

sum_err_step = 0
sum_err_evol = 0

step_calctime = list()
evol_calctime = list()

for i in tqdm(range(len(traindata.testpath))):
#     print(i+1,end='\r')
    testpath = traindata.testpath[i]

    if arg_type == 'CSE':
        err_test, err_evol, step_time, evol_time,n, n_hat, n_evol  = test.test_model(model, testpath, meta, printing = False)
    elif arg_type == 'L96':
        err_test, err_evol, step_time, evol_time,n, n_hat, n_evol  = test.test_model(model, N_L96, testpath, meta, printing = False)

    sum_err_step += err_test
    sum_err_evol += err_evol

    step_calctime.append(step_time)
    evol_calctime.append(evol_time)

utils.makeOutputDir(path+'/test')

np.save(path+ '/test/sum_err_step.npy', np.array(sum_err_step/len(traindata.testpath)))
np.save(path+ '/test/sum_err_evol.npy', np.array(sum_err_evol/len(traindata.testpath)))

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
