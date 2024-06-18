# -*- coding: utf-8 -*-

## Import dependent libraries
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

## Import libraries developed by this study
from WBModel import PRNNLayer, ConvLayer, ScaleLayer# RegionalPRNNLayer
from hydrodata import DataforIndividual
import hydroutils

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'
#np.warnings.filterwarnings('ignore')

working_path = r'D:\Code\GW_Optimized'
####################
#   Basin set up   #
####################
basin_id = '03015500' # The basin_id can be changed to any 8-digit basin id contained in the basin_list.txt

hydrodata = DataforIndividual(working_path, basin_id).load_data()

# Plot the data loaded for overview
fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7] = plt.subplots(nrows=7, ncols=1, sharex='row', figsize=(15, 18))

ax1.plot(hydrodata['prcp(mm/day)'])
ax2.plot(hydrodata['tmean(C)'])
ax3.plot(hydrodata['dayl(day)'])
ax4.plot(hydrodata['srad(W/m2)'])
ax5.plot(hydrodata['vp(Pa)'])
ax6.plot(hydrodata['flow(mm)'])
ax7.plot(hydrodata['GW(mm)'])


ax1.set_title(f"Basin {basin_id}")
ax1.set_ylabel("prcp(mm/day)")
ax2.set_ylabel("tmean(C)")
ax3.set_ylabel("dayl(day)")
ax4.set_ylabel("srad(W/m2)")
ax5.set_ylabel("vp(Pa)")
ax6.set_ylabel("flow(mm)")
ax7.set_ylabel("GW(mm)")


# plt.savefig(r'D:\Code\GRU_GW\results\adds\11\Plots\forcing_07068000_1.png')
# hydrodata.to_csv(r'D:\Code\GRU_GW\results\adds\11\Timeseries\07068000_1.csv')

#plt.savefig(r'D:\Physical_Process\groundwater\GRU\05362000\forcing012345_3.png')


plt.show()



####################
#  Period set up   #
####################

training_start = '1985-01-01'
training_end= '2009-09-30'

# The REAL evaluation period is from '2000-10-01', while the model needs one-year of data for spinning up the model
testing_start = '2009-09-01' 
testing_end= '2014-09-30'

# Split data set to training_set and testing_set
train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
test_set = hydrodata[hydrodata.index.isin(pd.date_range(testing_start, testing_end))]



print(f"The training data set is from {training_start} to {training_end}, with a shape of {train_set.shape}")
print(f"The testing data set is from {testing_start} to {testing_end}, with a shape of {test_set.shape}")

def generate_train_test(train_set, test_set, wrap_length):
    train_x_np = train_set.values[:, [0, 1, 2, 3, 4, 5]]
    train_y_np = train_set.values[:, -1:]
    test_x_np = test_set.values[:, [0, 1, 2, 3, 4, 5]]
    test_y_np = test_set.values[:, -1:]
    
    wrap_number_train = (train_set.shape[0]-wrap_length)//365 + 1 #train_set.shape[0]代表矩阵的行数
    
    train_x = np.empty(shape = (wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.empty(shape = (wrap_number_train, wrap_length, train_y_np.shape[1]))

    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)
    
    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i*365:(wrap_length+i*365), :]
        train_y[i, :, :] = train_y_np[i*365:(wrap_length+i*365), :]
             
    return train_x, train_y, test_x, test_y

wrap_length=2190 # It can be other values, but recommend this value should not be less than 5 years (1825 days).
train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, wrap_length=wrap_length)

print(f'The shape of train_x, train_y, test_x, and test_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_y.shape}, {test_x.shape}, and {test_y.shape}')

def create_model(input_shape, seed, num_filters, model_type='hybrid'):
    """Create a Keras model.
    -- input_shape: the shape of input, controlling the time sequence length of the P-RNN
    -- seed: the random seed for the weights initialization of the 1D-CNN layers
    -- num_filters: the number of filters for the 1D-CNN layer
    -- model_type: can be 'hybrid', 'physical', or 'common'
    """
    
    x_input = Input(shape=input_shape, name='Input') #input_shape就是指输入张量的shape。例如，input_dim=784，说明输入是一个784维的向量，这相当于一个一阶的张量，它的shape就是(784,)。因此，input_shape=(784,)。
    
    if model_type == 'hybrid':
        hydro_output = PRNNLayer(mode= 'normal', name='Hydro')(x_input)
        x_hydro = Concatenate(axis=-1, name='Concat')([x_input, hydro_output])#把原始数据和初步计算所得的Q整合到一起
        x_scale = ScaleLayer(name='Scale')(x_hydro)
        cnn_output = ConvLayer(filters=num_filters, kernel_size=10, padding='causal', seed=seed, name='Conv1')(x_scale)
        cnn_output = ConvLayer(filters=1, kernel_size=1, padding='causal', seed=seed, name='Conv2')(cnn_output)
        model = Model(x_input, cnn_output)
    
    elif model_type == 'physical':
        hydro_output = PRNNLayer(mode= 'normal', name='Hydro')(x_input)
        model = Model(x_input, hydro_output)
    
    elif model_type == 'common':
        cnn_output = ConvLayer(filters=num_filters, kernel_size=10, padding='causal', seed=seed, name='Conv1')(x_input)
        cnn_output = ConvLayer(filters=1, kernel_size=1, padding='causal', seed=seed, name='Conv2')(cnn_output)
        model = Model(x_input, cnn_output)
    
    return model

def train_model(model, train_x, train_y, ep_number, lrate, save_path):
    """Train a Keras model.
    -- model: the Keras model object
    -- train_x, train_y: the input and target for training the model
    -- ep_number: the maximum epoch number
    -- lrate: the initial learning rate
    -- save_path: where the model will be saved
    """
    
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)
    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)
    tnan = callbacks.TerminateOnNaN()

    model.compile(loss=hydroutils.nse_loss, metrics=[hydroutils.nse_metrics], optimizer=optimizers.Adam(lr=lrate))
    history = model.fit(train_x, train_y, epochs=ep_number, batch_size=10000, callbacks=[save, es, reduce, tnan])
    
    return history
    
def test_model(model, test_x, save_path):
    """Test a Keras model.
    -- model: the Keras model object
    -- test_x: the input for testing the model
    -- save_path: where the model was be saved
    """
    model.load_weights(save_path, by_name=True)
    pred_y = model.predict(test_x, batch_size=10000)
    
    return pred_y

Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_hybrid = f'{working_path}/results/{basin_id}_hybrid.h5'

model = create_model((train_x.shape[1], train_x.shape[2]), seed = 200, num_filters = 8, model_type='hybrid')
#x.shape[0]代表包含二维数组的个数，x.shape[1]表示二维数组的行数，x.shape[2]表示二维数组的列数。
model.summary()
hybrid_history = train_model(model, train_x, train_y, ep_number=400, lrate=0.01, save_path=save_path_hybrid)


Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_physical = f'{working_path}/results/{basin_id}_physical.h5'

model = create_model((train_x.shape[1], train_x.shape[2]), seed = 200, num_filters = 8, model_type='physical')
model.summary()
hybrid_history = train_model(model, train_x, train_y, ep_number=400, lrate=0.01, save_path=save_path_physical)



def normalize(data):
    data_mean = np.mean(data, axis=-2, keepdims=True)
    data_std = np.std(data, axis=-2, keepdims=True)
    data_scaled = (data - data_mean) / data_std
    return data_scaled, data_mean, data_std

Path(f'{working_path}/results').mkdir(parents=True, exist_ok=True)
save_path_common = f'{working_path}/results/{basin_id}_common.h5'

model = create_model((train_x.shape[1], train_x.shape[2]), seed = 200, num_filters = 8, model_type='common')
model.summary()

train_x_nor, train_x_mean, train_x_std = normalize(train_x)
train_y_nor, train_y_mean, train_y_std = normalize(train_y)

common_history = train_model(model, train_x_nor, train_y_nor,ep_number=400, lrate=0.01, save_path=save_path_common)

####################
#  Hybrid DL model #
####################
model = create_model((test_x.shape[1], test_x.shape[2]), seed = 200, num_filters = 8, model_type='hybrid')
flow_hybrid = test_model(model, test_x, save_path_hybrid)

####################
# Physical NN model#
####################
model = create_model((test_x.shape[1], test_x.shape[2]), seed = 200, num_filters = 8, model_type='physical')
flow_physical = test_model(model, test_x, save_path_physical)

####################
#  Common NN model #
####################
model = create_model((test_x.shape[1], test_x.shape[2]), seed = 200, num_filters = 8, model_type='common')
#We use the feature means/stds of the training period for normalization
test_x_nor = (test_x - train_x_mean) / train_x_std 

flow_common = test_model(model, test_x_nor, save_path_common)
#We use the feature means/stds of the training period for recovery
flow_common = flow_common * train_y_std + train_y_mean

#We can export these timeseries.

evaluate_set = test_set.loc[:, ['prcp(mm/day)','GW(mm)']]
evaluate_set['flow_obs'] = evaluate_set['GW(mm)']
evaluate_set['flow_hybrid'] = np.clip(flow_hybrid[0, :, :], a_min = 0, a_max = None)
evaluate_set['flow_physical'] = np.clip(flow_physical[0, :, :], a_min = 0, a_max = None)
evaluate_set['flow_common'] = np.clip(flow_common[0, :, :], a_min = 0, a_max = None)

def addYears(date, years):
    result = date + timedelta(366 * years)
    if years > 0:
        while result.year - date.year > years or date.month < result.month or date.day < result.day:
            result += timedelta(-1)
    elif years < 0:
        while result.year - date.year < years or date.month > result.month or date.day > result.day:
            result += timedelta(1)
    return result

evaluation_start = datetime.strftime(addYears(datetime.strptime(testing_start, '%Y-%m-%d'), 1), '%Y-%m-%d')
evaluation_end = testing_end

def calc_nse(y_true, y_pred):
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - numerator / denominator

# We only evaluate the data in the evaluation period
date_range = pd.date_range(evaluation_start, evaluation_end) 
evaluate_set = evaluate_set[evaluate_set.index.isin(date_range)]

# Calculate respective NSE values
nse_hybrid = calc_nse(evaluate_set['flow_obs'].values, evaluate_set['flow_hybrid'].values)
nse_physical = calc_nse(evaluate_set['flow_obs'].values, evaluate_set['flow_physical'].values)
nse_common = calc_nse(evaluate_set['flow_obs'].values, evaluate_set['flow_common'].values)

def evaluation_plot(ax, plot_set, plot_name, line_color, nse_values, basin_id):
    ax.plot(plot_set['flow_obs'], label="observation", color='black', ls='--')
    ax.plot(plot_set[plot_name], label="simulation", color=line_color, lw=1.5)
    ax.set_ylim([0,20])
    ax.set_title(f"Basin {basin_id} - Test set NSE: {nse_values:.3f}")
    ax.set_ylabel("GWflow(mm)")
    ax.legend()

plot_set = evaluate_set[evaluate_set.index.isin(pd.date_range(testing_start, testing_end))]

fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1, sharex='row', figsize=(15, 12))

evaluation_plot(ax1, plot_set, 'flow_hybrid', '#e41a1c', nse_hybrid, basin_id)
evaluation_plot(ax2, plot_set, 'flow_physical', '#377eb8', nse_physical, basin_id)
evaluation_plot(ax3, plot_set, 'flow_common', '#4daf4a', nse_common, basin_id)

ax1.annotate('Hybrid DL model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
ax2.annotate('Physical NN model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)
ax3.annotate('Common NN model', xy=(0.05, 0.9), xycoords='axes fraction', size=12)




# #plt.savefig(r'D:\Code\GW_Optimized\camels\plots\18\11264500.png')
# plt.savefig(r'D:\Physical_Process\groundwater\GRU\05362000\results012345_3.png')
plt.show()



# Q_observed = evaluate_set['flow_obs']
# df = pd.DataFrame(Q_observed)
# df.to_csv(r'D:\Physical_Process\groundwater\GRU\05362000\observed_NSE012345_3.csv')

# Q_observed = evaluate_set['flow_hybrid']
# df = pd.DataFrame(Q_observed)
# df.to_csv(r'D:\Physical_Process\groundwater\GRU\05362000\hybrid_NSE012345_3.csv')

# Q_observed = evaluate_set['flow_physical']
# df = pd.DataFrame(Q_observed)
# df.to_csv(r'D:\Physical_Process\groundwater\GRU\05362000\physical_NSE012345_3.csv')

# Q_observed = evaluate_set['flow_common']
# df = pd.DataFrame(Q_observed)
# df.to_csv(r'D:\Physical_Process\groundwater\GRU\05362000\common_NSE012345_3.csv')

# # df=plot_set['flow_hybrid']

# # df.to_csv(r'D:\DATA\test\tst.txt', header=None, sep='\t', index=False)


