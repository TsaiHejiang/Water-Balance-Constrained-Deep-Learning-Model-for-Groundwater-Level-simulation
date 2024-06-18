# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:32:04 2021

@author: TsaiHejiang
"""

from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf


class PRNNLayer(Layer):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    f: Rate of decline in flow from catchment bucket | Range: (0, 0.1)
    smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
    qmax: Maximum subsurface flow at full bucket     | Range: (10, 50)
    ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
    tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
    tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)
    """

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(PRNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.f = self.add_weight(name='f', shape=(1,),  #
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.Kc = self.add_weight(name='Kc', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value= 0.2, max_value=1.0, rate=0.9),
                                    trainable = True)
        self.SCmax = self.add_weight(name='SCmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1/15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.spmax = self.add_weight(name='spmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1/15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qpmax = self.add_weight(name='qpmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.kp = self.add_weight(name='kp', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value= 0.1, max_value=1.0, rate=0.9),
                                    trainable = True)
        self.sgmax = self.add_weight(name='sgmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1/15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qgmax = self.add_weight(name='qgmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)        
        self.Kl = self.add_weight(name='Kl', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value= 0.1, max_value=1.0, rate=0.9),
                                    trainable = True)
        self.Kn = self.add_weight(name='Kn', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value= 0.1, max_value=1.0, rate=0.9),
                                    trainable = True)

        super(PRNNLayer, self).build(input_shape)

    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (K.tanh(5 * x) + 1) / 2
    
    def canopybucket(self, s0, SCmax, dayl, p, Kc ) :
        
        """
        Equations for the canopybucket
        Sc(t) = Kc * Dc* LAI(t) or exp algorithm for non-linear relations
        if Day(L)>0.5:
            Kc = Kcs 
        else:
            Kc=Kcw
        if s0<0:
            Pfall = 0
        elseif s0>smax:
            Pfall = p
        else:
            Pfall = p-Kc*Dc*LAI(t)
            
        Using HUC:01548500 as an example:
        Dc : the coefficient of dominant land-cover fraction  
        LAI max:     
        LAI min: lai min = lai max - lai diff
        """
        """
          if dayl >= 0.5:
            SCmax = SCmax * 1.5 #(0.3 , 1.5)
            Lai = 4.78
            Kc = Kc * 1
          else:
            SCmax =SCmax * 0.6#(0.12, 0.6)
            Lai = 0.5
            Kc = Kc * 0.4
            
        """
        Kc = Kc * 0.5 #(0.1,0.5)
        SCmax = SCmax*1.5  #(0.3, 4.5)
        
        Dc = 0.986 * 0.862        
        SCmax = self.heaviside(dayl-0.5) * SCmax*2 + self.heaviside(0.5-dayl) * SCmax*0.6
        Lai = self.heaviside(dayl-0.5) * 0.478 + self.heaviside(0.5- dayl) * 0.15
        Kc = self.heaviside(dayl-0.5) * Kc*1 + self.heaviside(0.5-dayl) *Kc* 0.4
        
        Pintc = self.heaviside(p-0.1)*self.heaviside(s0) * self.heaviside(s0-SCmax)*p +\
                self.heaviside(p-0.1)*self.heaviside(SCmax - s0)*Kc * Dc* Lai 
        
        return Pintc    


    def rainsnowpartition(self, p, t, tmin):
        """
        Equations to partition incoming precipitation into rain or snow
            if t < tmin:
                psnow = p
                prain = 0
            else:
                psnow = 0
                prain = p
        """
        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    def snowbucket(self, s1, t, ddf, tmax):
        """
        Equations for the snow bucket
            if t > tmax:
                if s0 > 0:
                    melt = min(s0, ddf*(t - tmax))
                else:
                    melt = 0
            else:
                melt = 0
        """
        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s1) * K.minimum(s1, ddf * (t - tmax))

        return melt
    
    def preferentialbucket(self, s2, p, kp, spmax, qpmax,f):
        f = f / 10
        spmax = spmax * 1950 #(210, 3150)
         
        qpmax = qpmax * 40 #(8,40)
        
        qpref = self.heaviside(s2)*self.heaviside(s2-spmax)*qpmax + \
                self.heaviside(s2)*self.heaviside(spmax-s2)*kp*p*K.exp(-1 * f * (spmax - s2))
        
        return qpref
        

    def capillarybucket(self, s3, pet, f, smax, qmax):
        """
        Equations for the soil bucket
            if s1 < 0:
                et = 0
                qsub = 0
                qsurf = 0
            elif s1 > smax:
                et = pet
                qsub = qmax
                qsurf = s1 - smax
            else:
                et = pet * (s1 / smax)
                qsub = qmax * exp(-f * (smax - s1))
                qsurf = 0
        """
        f = f / 10  # (0, 0.1)
        smax = smax * 1950  # (210, 3150)
        qmax = qmax * 50 # (10, 50)

        et = self.heaviside(s3) * self.heaviside(s3 - smax) * pet + \
            self.heaviside(s3) * self.heaviside(smax - s3) * pet * (s3 / smax)
        qsub = self.heaviside(s3) * self.heaviside(s3 - smax) * qmax + \
            self.heaviside(s3) * self.heaviside(smax - s3) * qmax * K.exp(-1 * f * (smax - s3))
        qout = self.heaviside(s3) * self.heaviside(s3 - smax) * (s3 - smax)

        return [et, qsub, qout]
    
    def gravitybucket(self, s4, f, p, Kl, Kn, sgmax, qgmax):
        f = f / 10  # (0, 0.1)
        Kl = Kl * 0.5
        Kn = Kn * 0.5
        sgmax = sgmax * 1950  # (210, 3150)
        qgmax = qgmax * 40 # (10, 50)
        
        
        qslow = self.heaviside(s4)*self.heaviside(s4-sgmax)* qgmax + \
            self.heaviside(s4)*self.heaviside(sgmax-s4)*(p*Kl+p*p*Kn)* K.exp(-1 * f * (sgmax - s4))
        
        return qslow



    def step_do(self, step_in, states):  # Define step function for the RNN
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket
        s2 = states[0][:, 2:3]  # Preferential bucket
        s3 = states[0][:, 3:4]  # Capillary bucket
        s4 = states[0][:, 4:5]  # Gravity buckect
        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        dayl = step_in[:, 2:3]
        pet = step_in[:, 3:4]

        # Partition precipitation into rain and snow
        _pintc = self.canopybucket(s0, self.SCmax, dayl, p, self.Kc)
        
        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)
        # Snow bucket
        _m = self.snowbucket(s1, t, self.ddf, self.tmax)
        
        _qpref = self.preferentialbucket(s2, p, self.kp, self.spmax, self.qpmax, self.f)
        
        # Capillary bucket
        [_et, _qsub, _qout] = self.capillarybucket(s3, pet, self.f, self.smax, self.qmax)
        
        #Gravity bukect
        _qslow = self.gravitybucket(s4, self.f, p, self.Kl, self.Kn, self.sgmax, self.qgmax)


        # Water balance equations
        _ds0 = _pintc
        _ds1 = _ps - _m
        _ds2 = _qpref
        _ds3 = _pr + _m - _et - _qsub - _qout
        _ds4 = _qslow


        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)
        next_s2 = s2 + K.clip(_ds2, -1e5, 1e5)
        next_s3 = s3 + K.clip(_ds3, -1e5, 1e5)
        next_s4 = s4 + K.clip(_ds4, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1, next_s2, next_s3, next_s4], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, dayl, pet), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 5))]

        # Recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]
        s2 = outputs[:, :, 2:3]
        s3 = outputs[:, :, 3:4]
        s4 = outputs[:, :, 4:5]

        # Calculate final process variables
        pintc = self.canopybucket(s0, self.SCmax, dayl, prcp, self.Kc)
        m = self.snowbucket(s1, tmean, self.ddf, self.tmax)
        qpref = self.preferentialbucket(s2, prcp, self.kp, self.spmax, self.qpmax, self.f)
        [et, qsub, qout] = self.capillarybucket(s3, pet, self.f, self.smax, self.qmax)
        qslow = self.gravitybucket(s4, self.f, prcp, self.Kl, self.Kn, self.sgmax, self.qgmax)

        if self.mode == "normal":
            return qsub - qpref - qslow 
        elif self.mode == "analysis":
            return K.concatenate([s0, s1, s2, s3, s4, m, et, qsub, qout, pintc, qpref, qslow], axis=-1)
        

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 12)


class ConvLayer(Layer):
    """Implementation of the standard 1D-CNN layer

    Hyper-parameters (same as the Conv1D in https://keras.io/layers/convolutional/)
    ----------
    filters: The dimensionality of the output space (i.e. the number of output filters in the convolution)
    kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window
    padding: One of "valid", "causal" or "same"
    seed: Random seed for initialization
    """

    def __init__(self, filters, kernel_size, padding, seed=200, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seed = seed
        super(ConvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size, input_shape[-1], self.filters),
                                      initializer=initializers.random_uniform(seed=self.seed),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=initializers.Zeros(),
                                    trainable=True)

        super(ConvLayer, self).build(input_shape)

    def call(self, inputs):
        
        outputs = K.conv1d(inputs, self.kernel, strides=1, padding=self.padding)
        outputs = K.elu(K.bias_add(outputs, self.bias))
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)


class ScaleLayer(Layer):
    """
    Scale the inputs with the mean activation close to 0 and the standard deviation close to 1
    """

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):
        met = inputs[:, :, :-1]
        flow = inputs[:, :, -1:]

        self.met_center = K.mean(met, axis=-2, keepdims=True)
        self.met_scale = K.std(met, axis=-2, keepdims=True)
        self.met_scaled = (met - self.met_center) / self.met_scale

        return K.concatenate([self.met_scaled, flow], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
