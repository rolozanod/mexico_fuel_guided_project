import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

from modules.data import create_fuel_dataframe

def get_timeseries(data):

    price_pvt = data.set_index(['date', 'state', 'mun']).unstack(level=[1,2]).price
    litres_pvt = data.set_index(['date', 'state', 'mun']).unstack(level=[1,2]).litres

    day = 24*60*60

    week = 7*day

    month = 30*day

    year = 365.2425

    dates_feats = pd.DataFrame(litres_pvt.index).assign(
        day_sin = lambda c: c.date.dt.weekday.apply(lambda r: np.sin(r * (2 * np.pi / week))),
        day_cos = lambda c: c.date.dt.weekday.apply(lambda r: np.cos(r * (2 * np.pi / week))),
        week_sin = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.sin(r * (2 * np.pi / week))),
        week_cos = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.cos(r * (2 * np.pi / week))),
        month_sin = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.sin(r * (2 * np.pi / month))),
        month_cos = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.cos(r * (2 * np.pi / month))),
    ).set_index('date')

    print(f'Timesteps: {price_pvt.shape[0]}\nLocations: {price_pvt.shape[1]}')

    return price_pvt, litres_pvt, dates_feats

def sigmoid(x):
    return 1/(1+np.exp(-x))

class WindowGenerator():
    def __init__(self, input_width, label_width, batch,
                min_share=0.1, keep_above=0.9,
                label_columns=None):

        data = create_fuel_dataframe(min_share=min_share, keep_above=keep_above)

        p_df, l_df, d_df = get_timeseries(data)

        test_size=0.2
        shift=1

        # Store the raw data.
        self.p_df = p_df
        self.l_df = l_df
        self.d_df = d_df

        self.coords = data.assign(aux=1).groupby(['state', 'mun', 'lat', 'lon']).agg({'aux': 'sum'}).reset_index().drop(columns=["aux"])
        
        # Batch and test size
        self.batch = batch
        self.test_size = test_size

        # Features indices
        self.features_columns = d_df.columns

        # Work out the label column indices.
        self.label_columns = p_df.columns
        self.column_indices = {name: i for i, name in
                            enumerate(p_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.index_map = {it:k for k, it in d_df.reset_index().date.drop_duplicates().to_dict().items()}

    def __repr__(self):

        print(len(self.test), '\n')
        _n_window = 0
        _l_window = ["inputs", "targets"]
        for window in self.test:
            for e in window:
                print(f"Window entry: {_l_window[_n_window]}\n")
                for ee in e:
                    for eee in ee:
                        for eeee in eee:
                            print(f'Windows: {len(window)}')
                            print(f'Tuples: {len(e)}')
                            print(f'Batches: {len(ee)}')
                            print(f'Time: {len(eee)}')
                            print(f'Features: {len(eeee)}', '\n')
                            break
                        break
                _n_window += 1
            break

        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    
    def np_state_creator(self, data):
        '''
        This function takes the times series data and generates states from it

        @param data: dataframe
        @param timestamp: datetime, timestamp
        @param window_size: int, number of previous days
        '''

        state = [0]

        for i in range(self.input_width - 1):
            try:
                ns = sigmoid((data[i+1] / data[i]) - 1)
            except():
                tqdm.write(f"Error: \n     w+1: {data[i+1]}\n       w: {data[i]}")
                raise NotImplementedError
            state.append(ns)

        return np.array([state])
    
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def state_creator(self, input):
        return tf.numpy_function(self.np_state_creator, [self, input], tf.float32)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch,)

        ds = ds.map(self.split_window)

        return ds

    def train_split(self, df, test_size):
        train = df.shift(-test_size).iloc[:-test_size]
        test = df.iloc[test_size:]

        train = train.shift(-test_size).iloc[:-test_size]
        val = train.iloc[test_size:]

        return train, val, test

    def target_split(self, df, history_window):
        window_df = df.shift(-history_window).iloc[:-history_window]
        target_df = df.iloc[history_window:]

        return window_df, target_df

    def create_ts_ds(self, p_df, l_df, d_df, history_window, batch_size, shuffle=True):

        c_df = pd.concat([p_df, l_df, d_df], axis=1).reset_index()
        c_df['date'] = c_df.date.map(self.index_map)

        # Create a tf.data.Dataset from the dataframe and labels.
        ds = self.make_dataset(c_df)
        
        ds = ds.map(lambda train, tgt: (
                (
                    train[:,:,1:len(self.label_columns)+1],
                    train[:,:,len(self.label_columns)+1:-len(self.features_columns)],
                    train[:,:,-len(self.features_columns):]
                ),
                (
                    tgt[:,:,1:len(self.label_columns)+1],
                    tgt[:,:,len(self.label_columns)+1:-len(self.features_columns)]
                )
            )
        )
        
        if shuffle:
            ds = ds.shuffle(len(c_df))
            
        # ds = ds.batch(batch_size)
        
        return ds

    def get_ds(self, scope, test_size, history_window, batch_size):

        test_size = int(len(self.p_df)*test_size)

        p_train, p_val, p_test = self.train_split(self.p_df, test_size)
        l_train, l_val, l_test = self.train_split(self.l_df, test_size)        
        d_train, d_val, d_test = self.train_split(self.d_df, test_size)

        if scope == "train":
            ds = self.create_ts_ds(p_train, l_train, d_train, history_window=history_window, batch_size=batch_size)

        if scope == "validation":
            ds = self.create_ts_ds(p_val, l_val, d_val, history_window=history_window, batch_size=batch_size, shuffle=False)

        if scope == "test":
            ds = self.create_ts_ds(p_test, l_test, d_test, history_window=history_window, batch_size=batch_size, shuffle=False)

        return ds
    
    def test_data(self, date_idx, df_tuples = None):
        if df_tuples is None:
            inputs = (
                self.p_df.iloc[date_idx:date_idx+self.input_width].values.reshape(1, self.input_width, len(self.label_columns)),
                self.l_df.iloc[date_idx:date_idx+self.input_width].values.reshape(1, self.input_width, len(self.label_columns)),
                self.d_df.iloc[date_idx:date_idx+self.input_width].values.reshape(1, self.input_width, len(self.features_columns))
            )
            
            response = (
                self.p_df.iloc[date_idx+self.input_width:date_idx+self.input_width+1],
                self.l_df.iloc[date_idx+self.input_width:date_idx+self.input_width+1]
                    )

            return inputs, response

        else:
            p_df = df_tuples[0]
            l_df = df_tuples[1]
            d_df = df_tuples[2]

            inputs = (
                p_df.iloc[date_idx:date_idx+self.input_width].values.reshape(1, self.input_width, len(self.label_columns)),
                l_df.iloc[date_idx:date_idx+self.input_width].values.reshape(1, self.input_width, len(self.label_columns)),
                d_df.iloc[date_idx:date_idx+self.input_width].values.reshape(1, self.input_width, len(self.features_columns))
            )

            return inputs

    @property
    def train(self):
        return self.get_ds(scope='train', test_size=self.test_size, history_window=self.input_width, batch_size=self.batch)

    @property
    def val(self):
        return self.get_ds(scope='validation', test_size=self.test_size, history_window=self.input_width, batch_size=self.batch)

    @property
    def test(self):
        return self.get_ds(scope='test', test_size=self.test_size, history_window=self.input_width, batch_size=self.batch)

class TimeSeriesBlock(tf.keras.Model):
    def __init__(self, diff=True):
        super(TimeSeriesBlock, self).__init__()
        self.diff = diff

    def call(self, inputs):

        if self.diff:
            in_layer = tf.math.sigmoid(tf.math.divide_no_nan(inputs[:,slice(1, None),:], inputs[:,slice(0,-1),:])-1)

            mean = tf.math.reduce_mean(inputs, axis=1, keepdims=True)
            std = tf.math.reduce_std(inputs, axis=1, keepdims=True)

            return in_layer, mean, std
        
        else:
            in_layer = inputs[:,slice(1, None),:]

            return in_layer

class Sampling(tf.keras.layers.Layer):
    def __init__(self):
        super(Sampling, self).__init__()
        # self.random_normal_dimensions = random_normal_dimensions

    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5*sigma)*epsilon

class EncoderLayers(tf.keras.Model):
    def __init__(self, input_dims, latent_dims, filters, kernel_size, strides, h_units):
        super(EncoderLayers, self).__init__()
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.h_units = h_units

        # Define a Conv1D layers
        self.conv1D_0 = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            activation='relu',
            padding='same',
            name="encoder_conv1D_0")

        self.conv1D_1 = tf.keras.layers.Conv1D(
            filters=int(self.filters*2),
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            activation='relu',
            padding='same',
            name="encoder_conv1D_1")

        self.conv1D_2 = tf.keras.layers.Conv1D(
            filters=int(self.filters*4),
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            activation='relu',
            padding='same',
            name="encoder_conv1D_2")

        self.flatten = tf.keras.layers.Flatten(name='encoder_flatten')

        self.h_dense = tf.keras.layers.Dense(self.h_units, activation='relu', name="encoder_dense")
        self.m_dense = tf.keras.layers.Dense(self.latent_dims, name="latent_mu")
        self.s_dense = tf.keras.layers.Dense(self.latent_dims, name="latent_sigma")

    def call(self, inputs):
        
        x = self.conv1D_0(inputs)
        x = self.conv1D_1(x)
        x = self.conv1D_2(x)
        x = self.flatten(x)
        x = self.h_dense(x)

        mu = self.m_dense(x)
        sigma = self.s_dense(x)

        return mu, sigma

def EncoderModel(batch_size, input_width, input_dims, latent_dims, filters, kernel_size, strides, h_units):

        inputs = tf.keras.layers.Input(shape=(input_width, input_dims), batch_size=batch_size)

        encoder_ly = EncoderLayers(input_dims, latent_dims, filters, kernel_size, strides, h_units)

        mu, sigma = encoder_ly(inputs)

        samples = Sampling()((mu, sigma))

        model = tf.keras.Model(inputs, outputs=[mu, sigma, samples])

        return model

class DecoderLayers(tf.keras.Model):
    def __init__(self, i_units, input_width, input_dims, latent_dims, filters, kernel_size, strides, random=False, amplitude=1):
        super(DecoderLayers, self).__init__()
        self.input_dims = input_dims
        self.input_width = input_width
        self.latent_dims = latent_dims
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.i_units = i_units
        self.random = random
        self.amplitude = amplitude

        # Define a Conv1D layers
        self.conv1D_out = tf.keras.layers.Conv1DTranspose(
            filters=input_dims,
            kernel_size=(int(self.kernel_size/7),),
            strides=(1,),
            activation='linear',
            padding='valid',
            name="decoder_conv1D_out")

        self.conv1D_0 = tf.keras.layers.Conv1DTranspose(
            filters=self.filters,
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding='same',
            activation='relu',
            name="decoder_conv1D_0")

        self.conv1D_1 = tf.keras.layers.Conv1DTranspose(
            filters=int(self.filters*2),
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding='same',
            activation='relu',
            name="decoder_conv1D_1")

        self.conv1D_2 = tf.keras.layers.Conv1DTranspose(
            filters=int(self.filters*4),
            kernel_size=(self.kernel_size,),
            strides=(self.strides,),
            padding='same',
            activation='relu',
            name="decoder_conv1D_2")

        self.reshape = tf.keras.layers.Reshape((-1, self.latent_dims), name='decoder_reshape')

        self.i_dense = tf.keras.layers.Dense(self.i_units, activation='relu', name="decoder_dense")

        self.sample = tf.random.normal

    def call(self, inputs):
        
        if self.random:
            x = inputs + self.sample(shape=inputs.shape, mean=0, stddev=self.amplitude)#*tf.math.reduce_mean(inputs, axis=-1, keepdims=True)
        else:
            x = inputs

        x = self.i_dense(x)
        x = self.reshape(x)
        x = self.conv1D_2(x)
        x = self.conv1D_1(x)
        x = self.conv1D_0(x)
        x = self.conv1D_out(x)

        return x

def DecoderModel(batch_size, input_width, input_dims, latent_dims, filters, kernel_size, strides, random=False, amplitude=1):

        inputs = tf.keras.layers.Input(shape=(latent_dims), batch_size=batch_size)
        # features = tf.keras.layers.Input(shape=(feature_dims), batch_size=batch_size)

        decoder_ly = DecoderLayers(latent_dims*7, input_width, input_dims, latent_dims, filters, kernel_size, strides, random=random, amplitude=amplitude)

        outputs = decoder_ly(inputs)

        outputs_hist = outputs[:, slice(0,-1), :]

        outputs_pred = outputs[:, slice(-1,None), :]

        model = tf.keras.Model(inputs, outputs=[outputs_hist, outputs_pred])

        return model

def kl_reconstruction_loss(inputs, outputs, mu, sigma):
    """ Computes the Kullback-Leibler Divergence (KLD)
    Args:
        inputs -- batch from the dataset
        outputs -- output of the Sampling layer
        mu -- mean
        sigma -- standard deviation

    Returns:
        KLD loss
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    return tf.reduce_mean(kl_loss) * -0.5

def vae_model(encoder, decoder, batch_size, input_width, input_dims, latent_dims):
    """Defines the VAE model
    Args:
        encoder -- the encoder model
        decoder -- the decoder model
        input_shape -- shape of the dataset batch

    Returns:
        the complete VAE model
    """
    # set the inputs
    inputs = tf.keras.layers.Input(shape=(input_width, input_dims), batch_size=batch_size)
    
    # get mu, sigma, and z from the encoder output
    mu, sigma, z = encoder(inputs)

    # get reconstructed output from the decoder
    reconstructed, predicted = decoder(z)

    # define the inputs and outputs of the VAE
    model = tf.keras.Model(inputs=inputs, outputs=[reconstructed, predicted])

    # # add the KL loss
    loss = kl_reconstruction_loss(inputs, z, mu, sigma)
    model.add_loss(loss)

    return model

def get_models(batch_size, input_width, input_dims, latent_dims, filters, kernel_size, strides, h_units, random=False, amplitude=1):
    """Returns the encoder, decoder, and vae models"""
    ts = TimeSeriesBlock()
    encoder = EncoderModel(batch_size, input_width, input_dims, latent_dims, filters, kernel_size, strides, h_units)
    decoder = DecoderModel(batch_size, input_width, input_dims, latent_dims, filters, kernel_size, strides, random=random, amplitude=amplitude)
    vae = vae_model(encoder, decoder, batch_size, input_width, input_dims, latent_dims)
    return ts, encoder, decoder, vae

class MAPE_Revenue_Error(tf.keras.losses.Loss):
  
    #initialize instance attributes
    def __init__(self, tuples_history, section):
        super().__init__()
        self.section = section

        prices, litres = tuples_history
        revenue = tf.math.multiply(prices, litres)

        total_revenue = tf.reduce_sum(revenue, axis=2, keepdims=True)
        total_revenue = tf.reduce_sum(total_revenue, axis=1, keepdims=True)
        reduced_revenue = tf.reduce_sum(revenue, axis=1, keepdims=True)
        weights = tf.math.divide_no_nan(reduced_revenue, total_revenue)
        self.weights = weights

        def wgt_vars(price, litres):
            w_revenue = tf.reduce_sum(tf.multiply(price, litres), axis=2, keepdims=True)
            w_litres = tf.reduce_sum(litres, axis=2, keepdims=True)
            w_price = tf.math.divide_no_nan(w_revenue, w_litres)

            return w_price, w_litres

        self.wgt_vars = wgt_vars

        def MAPE(true, pred):
            return 100 * tf.math.reduce_mean(tf.math.abs(tf.math.divide_no_nan((true - pred), true)), axis=1, keepdims=True)
        self.MAPE = MAPE

    #compute loss
    def call(self, true_tuple, pred_tuple):
        prices_pred, litres_pred = pred_tuple
        prices_true, litres_true = true_tuple

        w_pred_price, w_pred_litres = self.wgt_vars(prices_pred, litres_pred)

        w_true_price, w_true_litres = self.wgt_vars(prices_true, litres_true)

        ##### MAPE #####
        w_price_loss = self.MAPE(w_true_price, w_pred_price)
        w_litres_loss = self.MAPE(w_true_litres, w_pred_litres)
        
        price_loss = self.weights*self.MAPE(prices_true, prices_pred)
        litres_loss = self.weights*self.MAPE(litres_true, litres_pred)
        ##### MAPE #####

        c_litres_loss = tf.reduce_sum(tf.concat([litres_loss, w_litres_loss], axis=2), axis=2, keepdims=True)
        c_price_loss = tf.reduce_sum(tf.concat([price_loss, w_price_loss], axis=2), axis=2, keepdims=True)
        
        if self.section == 'price':
            loss = c_price_loss
        elif self.section == 'volume':
            loss = c_litres_loss
        else:
            loss = tf.concat([c_litres_loss, c_price_loss], axis=2)

        loss = tf.nn.compute_average_loss(loss)
        
        return loss

def train(price_vae, price_ts, litres_vae, litres_ts, window, MAX_EPOCHS = 250, optimizer=None):

    loss_object = MAPE_Revenue_Error

    price_test_loss = tf.keras.metrics.Mean(name='price_test_loss')
    volume_test_loss = tf.keras.metrics.Mean(name='volume_test_loss')

    if optimizer is None:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.000)

    def compute_loss(ground_truth, predictions, tuples_history, section):
        local_loss_object = loss_object(tuples_history=tuples_history, section=section)
        loss = local_loss_object(ground_truth, predictions)
        return loss

    def train_step(inputs):
        labels, targets = inputs
        price_ly, litres_ly, dates_ly = labels
        price_tgt, litres_tgt = targets
        price_lbl, prices_mean, prices_std = price_ts(price_ly)
        litres_lbl, litres_mean, litres_std = litres_ts(litres_ly)

        with tf.GradientTape(persistent=True) as tape:
            reconstructed_price, predicted_price = price_vae(price_lbl)
            reconstructed_volume, predicted_volume = litres_vae(litres_lbl)

            predicted_price = price_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_price), axis=1, keepdims=True)*(1+predicted_price)
            predicted_volume = litres_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_volume), axis=1, keepdims=True)*(1+predicted_volume)

            price_loss = compute_loss((price_tgt, litres_tgt), (predicted_price, predicted_volume), tuples_history=(price_lbl, litres_lbl), section='price') + sum(price_vae.losses)

            volume_loss = compute_loss((price_tgt, litres_tgt), (predicted_price, predicted_volume), tuples_history=(price_lbl, litres_lbl), section='volume') + sum(litres_vae.losses)

        price_gradients = tape.gradient(price_loss, price_vae.trainable_variables)
        price_gradients = [tf.where(tf.math.is_nan(grad), 0.0, grad) for grad in price_gradients]
        # if tf.math.reduce_sum([tf.math.reduce_sum(tf.where(tf.math.is_nan(grad), 1, 0)) for grad in price_gradients]) == 0:
        optimizer.apply_gradients(zip(price_gradients, price_vae.trainable_variables))

        litres_gradients = tape.gradient(volume_loss, litres_vae.trainable_variables)
        litres_gradients = [tf.where(tf.math.is_nan(grad), 0.0, grad) for grad in litres_gradients]
        # if tf.math.reduce_sum([tf.math.reduce_sum(tf.where(tf.math.is_nan(grad), 1, 0)) for grad in litres_gradients]) == 0:
        optimizer.apply_gradients(zip(litres_gradients, litres_vae.trainable_variables))

        return price_loss, volume_loss

    #######################
    # Test Steps Functions
    #######################
    def test_step(inputs):
        labels, targets = inputs
        price_ly, litres_ly, dates_ly = labels
        price_tgt, litres_tgt = targets
        price_lbl, prices_mean, prices_std = price_ts(price_ly)
        litres_lbl, litres_mean, litres_std = litres_ts(litres_ly)

        reconstructed_price, predicted_price = price_vae(price_lbl)
        reconstructed_volume, predicted_volume = litres_vae(litres_lbl)

        predicted_price = price_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_price), axis=1, keepdims=True)*(1+predicted_price)
        predicted_volume = litres_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_volume), axis=1, keepdims=True)*(1+predicted_volume)

        price_loss_object = loss_object(tuples_history=(price_lbl, litres_lbl), section='price')
        price_loss = price_loss_object((price_tgt, litres_tgt), (predicted_price, predicted_volume))
        price_loss += price_vae.losses

        volume_loss_object = loss_object(tuples_history=(price_lbl, litres_lbl), section='volume')
        litres_loss = volume_loss_object((price_tgt, litres_tgt), (predicted_price, predicted_volume))
        litres_loss += litres_vae.losses

        price_test_loss.update_state(price_loss)
        volume_test_loss.update_state(litres_loss)

    # ## Training Loop
    # 
    # We can now start training the model.
    pbar = tqdm(range(MAX_EPOCHS))

    for epoch in pbar:
        # Do Training
        price_total_loss = 0.0
        litres_total_loss = 0.0
        num_batches = 0
        for batch in window.train:
            price_loss, volume_loss = train_step(batch)
            price_total_loss += price_loss
            litres_total_loss += volume_loss
            num_batches += 1
            price_train_loss = price_total_loss / num_batches
            litres_train_loss = litres_total_loss / num_batches

        # Do Testing
        for batch in window.val:
            test_step(batch)

        template = ("Epoch {:,.0f}, Price loss: {:,.4f}%, Litres loss: {:,.4f}%, Price val: {:,.4f}%, Litres val: {:,.4f}%")
        
        pbar.set_description(template.format(epoch+1, price_train_loss, litres_train_loss, price_test_loss.result(), volume_test_loss.result()))

        price_test_loss.reset_states()
        volume_test_loss.reset_states()

def save_weights(version, price_encoder, price_decoder, price_vae, volume_encoder, volume_decoder, volume_vae):
    price_encoder.save_weights(f"cpts/price_encoder_weights_{version}.h5")
    price_decoder.save_weights(f"cpts/price_decoder_weights_{version}.h5")
    price_vae.save_weights(f"cpts/price_vae_weights_{version}.h5")

    volume_encoder.save_weights(f"cpts/litres_encoder_weights_{version}.h5")
    volume_decoder.save_weights(f"cpts/litres_decoder_weights_{version}.h5")
    volume_vae.save_weights(f"cpts/litres_vae_weights_{version}.h5")

def load_weights(version, price_encoder, price_decoder, price_vae, volume_encoder, volume_decoder, volume_vae):
    price_encoder.load_weights(f"cpts/price_encoder_weights_{version}.h5")
    price_decoder.load_weights(f"cpts/price_decoder_weights_{version}.h5")
    price_vae.load_weights(f"cpts/price_vae_weights_{version}.h5")

    volume_encoder.load_weights(f"cpts/litres_encoder_weights_{version}.h5")
    volume_decoder.load_weights(f"cpts/litres_decoder_weights_{version}.h5")
    volume_vae.load_weights(f"cpts/litres_vae_weights_{version}.h5")

def run_backtest(price_vae, price_ts, litres_vae, litres_ts, window, plot=True):

    loss_object = MAPE_Revenue_Error

    price_test_loss = tf.keras.metrics.Mean(name='price_test_loss')
    volume_test_loss = tf.keras.metrics.Mean(name='volume_test_loss')

    #######################
    # Test Steps Functions
    #######################
    def test_step(inputs):
        labels, targets = inputs
        price_ly, litres_ly, dates_ly = labels
        price_tgt, litres_tgt = targets
        price_lbl, prices_mean, prices_std = price_ts(price_ly)
        litres_lbl, litres_mean, litres_std = litres_ts(litres_ly)

        reconstructed_price, predicted_price = price_vae(price_lbl)
        reconstructed_volume, predicted_volume = litres_vae(litres_lbl)

        predicted_price = price_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_price), axis=1, keepdims=True)*(1+predicted_price)
        predicted_volume = litres_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_volume), axis=1, keepdims=True)*(1+predicted_volume)

        price_loss_object = loss_object(tuples_history=(price_lbl, litres_lbl), section='price')
        price_loss = price_loss_object((price_tgt, litres_tgt), (predicted_price, predicted_volume))
        price_loss += price_vae.losses

        volume_loss_object = loss_object(tuples_history=(price_lbl, litres_lbl), section='volume')
        litres_loss = volume_loss_object((price_tgt, litres_tgt), (predicted_price, predicted_volume))
        litres_loss += litres_vae.losses

        price_test_loss.update_state(price_loss)
        volume_test_loss.update_state(litres_loss)

        return predicted_price, predicted_volume, price_tgt, litres_tgt

    predicted_price_ls = []
    predicted_volume_ls = []
    price_tgt_ls = []
    litres_tgt_ls = []

    # Do Testing
    for batch in window.test:
        predicted_price, predicted_volume, price_tgt, litres_tgt = test_step(batch)
        
        predicted_price_ls.append(predicted_price)
        predicted_volume_ls.append(predicted_volume)
        
        price_tgt_ls.append(price_tgt)
        litres_tgt_ls.append(litres_tgt)
    
    print(f'Price loss: {price_test_loss.result()}')
    
    print(f'Litres loss: {volume_test_loss.result()}')

    entries = predicted_price_ls[0][0].shape[1]

    def tf2df(tensor):
        return pd.DataFrame(np.concatenate([e.numpy()[-1,:].reshape(-1, entries) for e in tensor]))

    price_tgt_ls = tf2df(price_tgt_ls)
    predicted_price_ls = tf2df(predicted_price_ls)

    litres_tgt_ls = tf2df(litres_tgt_ls)
    predicted_volume_ls = tf2df(predicted_volume_ls)

    price_tgt_ls *= litres_tgt_ls
    predicted_price_ls *= predicted_volume_ls

    litres_tgt_ls = litres_tgt_ls.sum(axis=1)
    predicted_volume_ls = predicted_volume_ls.sum(axis=1)

    price_tgt_ls = price_tgt_ls.sum(axis=1).div(litres_tgt_ls)
    predicted_price_ls = predicted_price_ls.sum(axis=1).div(predicted_volume_ls)

    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(48, 12))
        
        price_tgt_ls.plot(ax=axes[0], label='Obs')
        
        predicted_price_ls.plot(ax=axes[0], label='Pred')
        axes[0].set_title('Price (MXN/L)')
        axes[0].legend()


        litres_tgt_ls.plot(ax=axes[1], label='Obs')
        predicted_volume_ls.plot(ax=axes[1], label='Pred')
        axes[1].set_title('Litres (L)')
        axes[1].legend()

        fig.show()

def test_df(price_vae, price_ts, litres_vae, litres_ts, window, idx, df_tuples = None):
    
    if df_tuples is None:
        labels, resp_df = window.test_data(idx, df_tuples)
    else:
        labels = window.test_data(idx, df_tuples)

    price_ly, litres_ly, dates_ly = labels
    price_lbl, prices_mean, prices_std = price_ts(price_ly)
    litres_lbl, litres_mean, litres_std = litres_ts(litres_ly)

    reconstructed_price, predicted_price = price_vae(price_lbl)
    reconstructed_volume, predicted_volume = litres_vae(litres_lbl)

    predicted_price = price_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_price), axis=1, keepdims=True)*(1+predicted_price)
    predicted_volume = litres_ly[:, slice(0,1), :]*tf.math.reduce_prod((1+reconstructed_volume), axis=1, keepdims=True)*(1+predicted_volume)

    if df_tuples is None:
        price_df = pd.concat(
            [
                pd.DataFrame(predicted_price.numpy()[0], columns=resp_df[0].columns),
                resp_df[0]
            ], axis = 0
        )
        litres_df = pd.concat(
            [
                pd.DataFrame(predicted_volume.numpy()[0], columns=resp_df[1].columns),
                resp_df[1]
            ], axis = 0
        )
    else:
        price_df = pd.DataFrame(predicted_price.numpy()[0], columns=window.label_columns)
        litres_df = pd.DataFrame(predicted_volume.numpy()[0], columns=window.label_columns)
    return price_df, litres_df

def forecast(price_vae, price_ts, litres_vae, litres_ts, window, init_date, final_date, plot=True, p_pbar=None, normalize=False, saturate=False):

    last_dt = window.d_df.index.max()
    last_idx = len(window.d_df)

    min_date = min(datetime.strptime(init_date, '%Y-%m-%d') - relativedelta(days=window.input_width), last_dt)

    all_dates = pd.date_range(start=min_date, end=final_date)
    existing_dates = pd.date_range(start=min_date, end=last_dt)
    # dates2fcst = pd.date_range(start=last_dt + relativedelta(days=1), end=final_date)
    dates2fcst = pd.date_range(start=init_date, end=final_date)

    p_fcst_df = window.p_df.loc[existing_dates].copy()
    t_fcst_df = window.l_df.loc[existing_dates].copy()
    d_fcst_df = window.d_df.loc[existing_dates].copy()

    if p_pbar is None:
        pbar = tqdm(dates2fcst)
    else:
        pbar = dates2fcst

    day = 24*60*60

    week = 7*day

    month = 30*day

    year = 365.2425

    for fcst_dt in pbar:            
        if p_pbar is None:
            pbar.set_description(f"Date: {fcst_dt}")
        else:
            p_pbar.set_description(f"Date: {fcst_dt}")

        p_fcst_resp_df, t_fcst_resp_df = test_df(price_vae, price_ts, litres_vae, litres_ts, window, idx=0,
            df_tuples = (
                p_fcst_df.loc[fcst_dt - relativedelta(days=window.input_width):fcst_dt],
                t_fcst_df.loc[fcst_dt - relativedelta(days=window.input_width):fcst_dt],
                d_fcst_df.loc[fcst_dt - relativedelta(days=window.input_width):fcst_dt]
            )
        )

        d_fcst_resp_df = pd.DataFrame([fcst_dt], columns = ['date']).assign(
            day_sin = lambda c: c.date.dt.weekday.apply(lambda r: np.sin(r * (2 * np.pi / week))),
            day_cos = lambda c: c.date.dt.weekday.apply(lambda r: np.cos(r * (2 * np.pi / week))),
            week_sin = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.sin(r * (2 * np.pi / week))),
            week_cos = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.cos(r * (2 * np.pi / week))),
            month_sin = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.sin(r * (2 * np.pi / month))),
            month_cos = lambda c: c.date.map(datetime.timestamp).apply(lambda r: np.cos(r * (2 * np.pi / month))),
        ).set_index('date')

        p_fcst_resp_df.index = [fcst_dt]
        t_fcst_resp_df.index = [fcst_dt]

        if normalize:
            normalize2mbbl = np.random.normal(loc=870, scale=30)
            # 158.98730272810

            t_fcst_resp_df *= normalize2mbbl/(t_fcst_resp_df.sum(axis=1).values[0]/158.98730272810/1000)

        if saturate:
            first_obs_prices = p_fcst_df.loc[fcst_dt - relativedelta(days=window.input_width):fcst_dt - relativedelta(days=window.input_width)]

            # limit annual compounded growth to pct
            pct = 0.09

            dub = ((1+pct)**(1/365)-1) # daily upper bound

            ub = ((1+dub)**56) - 1 # period upper bound

            rate = p_fcst_resp_df.div(first_obs_prices.values) - 1

            rate.clip(lower = -ub, upper=ub, inplace=True)

            p_fcst_resp_df = first_obs_prices*(1+rate.values)

            p_fcst_resp_df.index = [fcst_dt]

        if fcst_dt in d_fcst_df.index:
            p_fcst_df.loc[p_fcst_df.index == fcst_dt] = p_fcst_resp_df
            t_fcst_df.loc[t_fcst_df.index == fcst_dt] = t_fcst_resp_df
        else:
            p_fcst_df = pd.concat([p_fcst_df, p_fcst_resp_df], axis=0)
            t_fcst_df = pd.concat([t_fcst_df, t_fcst_resp_df], axis=0)
            d_fcst_df = pd.concat([d_fcst_df, d_fcst_resp_df], axis=0)

    p_fcst_df = p_fcst_df.loc[init_date:final_date].unstack().reset_index().rename(columns={'level_2': 'date', 0: 'price'})
    t_fcst_df = t_fcst_df.loc[init_date:final_date].unstack().reset_index().rename(columns={'level_2': 'date', 0: 'litres'})

    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(48, 12))
        
        m = p_fcst_df.groupby(["date"]).agg({'price': 'mean'})
        m.plot(ax=axes[0], label='Obs')
        axes[0].set_title('Price (MXN/L)')
        axes[0].legend()


        m = t_fcst_df.groupby(["date"]).agg({'litres': 'sum'})
        m.plot(ax=axes[1], label='Obs')
        axes[1].set_title('Litres (L)')
        axes[1].legend()

        fig.show()

    return p_fcst_df, t_fcst_df

# END