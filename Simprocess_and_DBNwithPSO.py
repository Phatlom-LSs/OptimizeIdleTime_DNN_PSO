import simpy
import numpy as np
import pandas as pd
from keras import Sequential
from sklearn.model_selection import train_test_split
from pyswarm import pso
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import keras
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from currency_converter import CurrencyConverter

df = pd.read_csv('C:/Users/763073/OneDrive - Seagate Technology/Desktop/SDET_UPH.csv')

ALLUM_production_time = {
    'EQUIP_ID': df['EQUIP_ID'],
    'Total_SETS': 1240,
    'SETS_per_Tray': 20,
    'Tray_per_batch': 10,
    'production_rate_trays_per_hour': df['UPH'],
    'batch_interval_sec': 10,
}

ET_production_time = {
    'TSTR_PART_NUM': df['TSTR_PART_NUM'],
    'bacth_processing_rate_per_machine': 1/3 * ALLUM_production_time['production_rate_trays_per_hour'],
    'UPH_ET': df['UPH2']
}

ALLUM_CAP_PER_DAY_PER_MC = {
    'TOTAL_SETS': 1240,
    'SETS_per_Tray': 20,
    'Tray_per_batch': 10,
    'Tray_per_MC': 62,
    'Cap_per_day_per_mc': 14200
}


print(ALLUM_production_time['production_rate_trays_per_hour'])
print(ET_production_time['UPH_ET'])




class SDET:
    def __init__(self, env, hpu_alm, spu_et, sets_qty):
        self.env = env
        self.hpu_alm = hpu_alm
        self.spu_et = spu_et
        self.sets_qty = sets_qty
        self.total_production_allum = 0
        self.total_production_et = 0
        self.run_result = []
    
    def allum_process(self, sets_qty, batch_size, alm_rate):
        process_time = ((sets_qty * batch_size) * (3600 / alm_rate))
        yield self.env.timeout(process_time)
        self.total_production_allum += batch_size
        print(f'Allum process time of batch size: {batch_size} trays at {self.env.now:.2f} seconds')
        return process_time, self.total_production_allum
    
    def et_process(self, sets_qty, batch_size, et_rate):
        yield self.env.timeout(10)
        if 6 <= batch_size <= 7:
            process_time = ((sets_qty * 2) * (3600 / et_rate)) + 22 # 6 trays + Purge Time
        else:
            process_time = (((sets_qty * 2) * 2) * (3600 / et_rate)) + 22 # 10 trays + Purge Time
    
        yield self.env.timeout(process_time)
        self.total_production_et += batch_size
        print(f'ET process time of batch size: {batch_size} trays at {self.env.now:.2f} seconds')
        return process_time, self.total_production_et
    
    def sim(self, sets_qty, batch_size, alm_rate, et_rate):
        print(f'Starting simulation for SETS Qty: {sets_qty}, Batch Size: {batch_size}')
        # ALLUM Start
        start_allum = self.env.now
        yield self.env.process(self.allum_process(sets_qty, batch_size, alm_rate))
        end_allum = self.env.now
        allum_time = end_allum - start_allum
        move_time_start = self.env.now

        yield self.env.timeout(22) # move time forward by 22 seconds to allow for ET machines to start processing
        move_time_end = self.env.now
        move_time = move_time_end - move_time_start
        allum_time += move_time

        # ET Start
        et_start = self.env.now
        yield self.env.process(self.et_process(sets_qty, batch_size, et_rate))
        et_end = self.env.now
        et_time = et_end - et_start

        idle_time = max(0, et_time - allum_time) 




        print(f'Simulation completed at {self.env.now:.2f} seconds')
        print(f'Idle Time : {idle_time:.2f}')

        self.run_result.append({
            'SETS Qty': sets_qty,
            'Batch Size': batch_size,
            'Idle Time': idle_time,
            'Total Production Allum': self.total_production_allum,
            'Total Production ET': self.total_production_et
        })
    


def run_sim(sets_qty, batch_size, alm_rate, et_rate):
    env = simpy.Environment()

    sdet_line = SDET(env, ALLUM_production_time['production_rate_trays_per_hour'], ET_production_time['UPH_ET'], sets_qty)
    env.process(sdet_line.sim(sets_qty, batch_size, alm_rate, et_rate))
    env.run(until=3600 * 24 * 7) # Simulate for 7 days

    return sdet_line.run_result



sets_qty_option = range(18, 21)
batch_size_option = [6, 10]

all_results = []

alm_rates = df['UPH'].tolist()
et_rates = df['UPH2'].tolist()

for sets_qty in sets_qty_option:
    for batch_size in batch_size_option:
        for alm_rate, et_rate in zip(alm_rates, et_rates):
            results = run_sim(sets_qty, batch_size, alm_rate, et_rate)
            all_results.append(results)

df_result = pd.DataFrame(columns=['SETS Qty', 'Batch Size', 'Idle Time', 'Total Production Allum', 'Total Production ET'])

for result in all_results:
    for res in result:
       print(f'SETS Qty: {res["SETS Qty"]}, Batch Size: {res["Batch Size"]}, Idle Time: {res["Idle Time"]}, Total Production Allum: {res["Total Production Allum"]}, Total Production ET: {res["Total Production ET"]}')
       df_result = pd.concat([df_result, pd.DataFrame([res])], ignore_index=True)
       if res['SETS Qty'] == 20 and res['Batch Size'] == 10:
           print(f'Idle Time at Current Standard {res["Idle Time"]}')


X = df_result[['SETS Qty', 'Batch Size']] # Features
y = df_result['Idle Time'] # Target



X = X.to_numpy()
y = y.to_numpy()

def prepare_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepare_data(X, y)
model = LinearRegression()
model.fit(X_train, y_train)
coeff = model.coef_
intercept = model.intercept_

print(f'Coefficients: {coeff}')
print(f'Intercept: {intercept}')

def VIF(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data
x_train_df = pd.DataFrame(X_train, columns=['SETS Qty', 'Batch Size'])
vif_data = VIF(x_train_df)
print(vif_data)


# Deep Neural Network
def DNN(input_dim, hidden_layers):
    model = Sequential()

    # Input layer
    model.add(keras.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(units, activation='leaky_relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(units, activation='leaky_relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.1))


    # Output layer
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

class PlotLearning(keras.callbacks.Callback):

    # initialize the class
    def __init__(self):
        super().__init__()

        self.losses = []
        self.val_losses = []
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax1 = plt.subplots(figsize=(12, 6))
        self.line1, = self.ax1.plot([], [], label='train loss')
        self.line2, = self.ax1.plot([], [], label='val loss')
        self.ax1.legend()
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Model Loss')
    # start the plot
    def on_train_begin(self, logs=None):
        pass

    # update the plot
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.line1.set_data(range(len(self.losses)), self.losses)
        self.line2.set_data(range(len(self.val_losses)), self.val_losses)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def on_train_end(self, logs=None):
        plt.ioff()
        plt.show()


def idle_time_cal(sets_qty, batch_size, alm_rate, et_rate):
    if 6 <= batch_size <= 7:
        idle_time = abs((batch_size * sets_qty * (3600 / alm_rate)) - (((sets_qty * 2) * (3600 / et_rate)))) + 22
    else:
        idle_time = abs((batch_size * sets_qty * (3600 / alm_rate)) - (((sets_qty * 2) * 2) * (3600 / et_rate))) + 22
 
    return idle_time

def objective_function(x, alm_rates, et_rates):
    sets_qty, batch_size = x
    if x[0] <= 16 or x[0] > 22:
        return float('inf')
    
    total_idle_time = 0
    for alm_rate, et_rate in zip(alm_rates, et_rates):
        total_idle_time += idle_time_cal(sets_qty, batch_size, alm_rate, et_rate)
    
    return total_idle_time

def rastrigin(x):
    n = len(x)
    A = 10
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def PSO(alm_rates, et_rates):
    lb = [19, 6]
    ub = [21, 10]

    xopt, fopt = pso(objective_function, lb, ub, args=(alm_rates, et_rates), swarmsize=1000, maxiter=500, debug=True
                    ,omega=0.7, phip=1.8, phig=1.8)
    return xopt, fopt

def train_test_eva(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    input_dim = X_train.shape[1]
    hidden_layers, _ = PSO(alm_rates, et_rates)

    hidden_layers = [int(unit) for unit in hidden_layers]
    print(f'Check {hidden_layers}') 
 
    model = DNN(input_dim, hidden_layers)

    reduce_Lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    eary_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=30, verbose=1, restore_best_weights=True)

    Plot_Learning = PlotLearning()

    tf.profiler.experimental.start('logdir')

    model.fit(X_train, y_train, epochs=500, batch_size=128, validation_data=(X_test, y_test),
              callbacks=[reduce_Lr, eary_stopping, Plot_Learning], validation_split=0.2)

    tf.profiler.experimental.stop()

    loss = model.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    optimal_sets_qty = hidden_layers[0]
    optimal_batch_size = hidden_layers[1]
    optimal_tray = 52 + optimal_batch_size

    print(f'Optimal SETS Qty: {optimal_sets_qty}, Optimal Batch Size: {optimal_batch_size}')
    print(f'Optimal Tray: {optimal_tray} from 62 Tray')
    print(f'Current Idle Time: {abs((10 * 20 * (3600 / alm_rate)) - (80 * (3600 / et_rate))) + 22}')
    if optimal_batch_size == 6:
        print(f'Optimal Idle Time: {abs((optimal_batch_size * optimal_sets_qty * (3600 / alm_rate)) - ((optimal_sets_qty * 2) * (3600 / et_rate))) + 22}')
    else:
        print(f'Optimal Idle Time: {abs((optimal_batch_size * optimal_sets_qty * (3600 / alm_rate)) - ((optimal_sets_qty * 4) * (3600 / et_rate))) + 22}')
    print(f'ALLUM Production Rate: {alm_rate} UPH, ET Production Rate: {et_rate} UPH')
    print(f'Delta Idle Time: {abs(((10 * 20 * (3600 / alm_rate)) - ((80 * (3600 / et_rate)))) + 22) - (abs((optimal_batch_size * optimal_sets_qty * (3600 / alm_rate)) - (((optimal_sets_qty * 2) * (3600 / et_rate)))) + 22)}')
    
    converter = CurrencyConverter()
    current_cost_usd = 28
    current_cost_all = 1240 * 1.4
    optimal_cost_usd = optimal_tray * optimal_sets_qty * 1.4
    cost_delta_usd = current_cost_all - optimal_cost_usd
    current_cost_thb = converter.convert(current_cost_all, 'USD', 'THB')
    optimal_cost_thb = converter.convert(optimal_cost_usd, 'USD', 'THB')
    cost_delta_thb = current_cost_thb - optimal_cost_thb

    print(f'Current Cost per SETs/ALLUM (1.4 USD/SETs): {current_cost_usd} USD or {current_cost_thb:.2f} Bath : All SETs use per ALLUM 62 Tray or 1,240 SETs cost = {1240 * 1.4} USD or {converter.convert(1240 * 1.4, "USD", "THB")} Bath') 
    print(f'Optimal Cost per SETs/ALLUM (1.4 USD/SETs): {(optimal_cost_usd):.2f} USD or {optimal_cost_thb:.2f} Bath : Optimal All SDET Lines SETs cost = {((optimal_tray * optimal_sets_qty) * 1.4):.2f} USD or {converter.convert((optimal_tray * optimal_sets_qty) * 1.4, "USD", "THB")} Bath')
    print(f'Delta Cost per SETs/ALLUM USD: {cost_delta_usd:.2f} USD and THB: {cost_delta_thb:.2f} THB')

train_test_eva(X, y)