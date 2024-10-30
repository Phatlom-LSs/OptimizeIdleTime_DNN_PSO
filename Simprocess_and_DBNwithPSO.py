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

df = pd.read_csv('File_Directory')

MC1_production_time = {
    'EQUIP_ID': df['EQUIP_ID'],
    'Total_SETS': 2000,
    'SETS_per_Tray': 100,
    'Tray_per_batch': 100,
    'production_rate_per_hour': df['UPH'],
    'batch_interval_sec': 100,
}

MC2_production_time = {
    'TSTR_PART_NUM': df['PART_NUM'],
    'bacth_processing_rate_per_machine': 1/3 * ALLUM_production_time['production_trays_per_hour'],
    'UPH_ET': df['UPH2']
}

MC1_CAP_PER_DAY_PER_MC = {
    'TOTAL_SETS': 2000,
    'SETS_per_Tray': 200,
    'Tray_per_batch': 100,
    'Tray_per_MC': 6200,
    'Cap_per_day_per_mc': 14200
}


print(MC1_production_time['production_rate_trays_per_hour'])
print(MC2_production_time['UPH_MC2'])




class Line:
    def __init__(self, env, hpu_mc1, spu_mc2, mat_qty):
        self.env = env
        self.hpu_mc1 = hpu_mc1
        self.spu_mc2 = spu_mc2
        self.mat_qty = mat_qty
        self.total_production_mc1 = 0
        self.total_production_mc2 = 0
        self.run_result = []
    
    def mc1_process(self, mat_qty, batch_size, mc1_rate):
        process_time = ((mat_qty * batch_size) * (3600 / mc1_rate))
        yield self.env.timeout(process_time)
        self.total_production_mc1 += batch_size
        print(f'mc1 process time of batch size: {batch_size} trays at {self.env.now:.2f} seconds')
        return process_time, self.total_production_mc1
    
    def mc2_process(self, mat_qty, batch_size, mc2_rate):
        yield self.env.timeout(10)
        if 6 <= batch_size <= 7:
            process_time = ((mat_qty * 2) * (3600 / mc2_rate)) + 22 # 6 trays + Purge Time
        else:
            process_time = (((mat_qty * 2) * 2) * (3600 / mc2_rate)) + 22 # 10 trays + Purge Time
    
        yield self.env.timeout(process_time)
        self.total_production_mc2 += batch_size
        print(f'mc2 process time of batch size: {batch_size} trays at {self.env.now:.2f} seconds')
        return process_time, self.total_production_mc2
    
    def sim(self, mat_qty, batch_size, mc1_rate, mc2_rate):
        print(f'Starting simulation for Mat Qty: {sets_qty}, Batch Size: {batch_size}')
        # ALLUM Start
        start_mc1 = self.env.now
        yield self.env.process(self.mc1_process(mat_qty, batch_size, mc1_rate))
        end_mc1 = self.env.now
        mc1_time = end_mc1 - start_mc1
        move_time_start = self.env.now

        yield self.env.timeout(22) # move time forward by 22 seconds to allow for ET machines to start processing
        move_time_end = self.env.now
        move_time = move_time_end - move_time_start
        mc1_time += move_time

        # ET Start
        mc2_start = self.env.now
        yield self.env.process(self.et_process(sets_qty, batch_size, et_rate))
        mc2_end = self.env.now
        mc2_time = mc2_end - mc2_start

        idle_time = max(0, mc2_time - mc1_time) 




        print(f'Simulation completed at {self.env.now:.2f} seconds')
        print(f'Idle Time : {idle_time:.2f}')

        self.run_result.append({
            'Mat Qty': mat_qty,
            'Batch Size': batch_size,
            'Idle Time': idle_time,
            'Total Production MC1': self.total_production_mc1,
            'Total Production MC2': self.total_production_mc2
        })
    


def run_sim(mat_qty, batch_size, mc1_rate, mc2_rate):
    env = simpy.Environment()

    process_line = SDET(env, MC1_production_time['production_rate_trays_per_hour'], MC2_production_time['UPH_ET'], mat_qty)
    env.process(process_line.sim(mat_qty, batch_size, mc1_rate, mc2_rate))
    env.run(until=3600 * 24 * 7) # Simulate for 7 days

    return process_line.run_result



mat_qty_option = range(18, 21)
batch_size_option = [6, 10]

all_results = []

mc1_rates = df['UPH'].tolist()
mc2_rates = df['UPH2'].tolist()

for mat_qty in mat_qty_option:
    for batch_size in batch_size_option:
        for mc1_rate, mc2_rate in zip(mc1_rates, mc2_rates):
            results = run_sim(mat_qty, batch_size, mc1_rate, mc2_rate)
            all_results.append(results)

df_result = pd.DataFrame(columns=['Mat Qty', 'Batch Size', 'Idle Time', 'Total Production MC1', 'Total Production MC2'])

for result in all_results:
    for res in result:
       print(f'Mat Qty: {res["Mat Qty"]}, Batch Size: {res["Batch Size"]}, Idle Time: {res["Idle Time"]}, Total Production MC1: {res["Total Production MC1"]}, Total Production MC2: {res["Total Production MC2"]}')
       df_result = pd.concat([df_result, pd.DataFrame([res])], ignore_index=True)
       if res['Mat Qty'] == 20 and res['Batch Size'] == 10:
           print(f'Idle Time at Current Standard {res["Idle Time"]}')


X = df_result[['Mat Qty', 'Batch Size']] # Features
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
x_train_df = pd.DataFrame(X_train, columns=['Mat Qty', 'Batch Size'])
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


def idle_time_cal(mat_qty, batch_size, mc1_rate, mc2_rate):
    if 6 <= batch_size <= 7:
        idle_time = abs((batch_size * mat_qty * (3600 / mc1_rate)) - (((mat_qty * 2) * (3600 / mc2_rate)))) + 22
    else:
        idle_time = abs((batch_size * mat_qty * (3600 / mc1_rate)) - (((mat_qty * 2) * 2) * (3600 / mc2_rate))) + 22
 
    return idle_time

def objective_function(x, mc1_rates, mc2_rates):
    mat_qty, batch_size = x
    if x[0] <= 16 or x[0] > 22:
        return float('inf')
    
    total_idle_time = 0
    for mc1_rate, mc2_rate in zip(mc1_rates, mc2_rates):
        total_idle_time += idle_time_cal(mat_qty, batch_size, mc1_rate, mc2_rate)
    
    return total_idle_time

def rastrigin(x):
    n = len(x)
    A = 10
    return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def PSO(mc1_rates, mc2_rates):
    lb = [19, 6]
    ub = [21, 10]

    xopt, fopt = pso(objective_function, lb, ub, args=(mc1_rates, mc2_rates), swarmsize=1000, maxiter=500, debug=True
                    ,omega=0.7, phip=1.8, phig=1.8)
    return xopt, fopt

def train_test_eva(X, y):
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    input_dim = X_train.shape[1]
    hidden_layers, _ = PSO(mc2_rates, mc2_rates)

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
    optimal_mat_qty = hidden_layers[0]
    optimal_batch_size = hidden_layers[1]
    optimal_box = 52 + optimal_batch_size

    print(f'Optimal Mat Qty: {optimal_sets_qty}, Optimal Batch Size: {optimal_batch_size}')
    print(f'Optimal Box: {optimal_box} from ?? Box')
    print(f'Current Idle Time: {abs((100 * 200 * (3600 / mc1_rate)) - (800 * (3600 / mc2_rate))) + 1}')
    if optimal_batch_size == 6:
        print(f'Optimal Idle Time: {abs((optimal_batch_size * optimal_mat_qty * (3600 / mc1_rate)) - ((optimal_mat_qty * 200) * (3600 / mc2_rate))) + 1}')
    else:
        print(f'Optimal Idle Time: {abs((optimal_batch_size * optimal_mat_qty * (3600 / mc1_rate)) - ((optimal_mat_qty * 4000) * (3600 / mc2_rate))) + 22}')
    print(f'MC1 Production Rate: {mc1_rate} UPH, MC2 Production Rate: {mc2_rate} UPH')
    print(f'Delta Idle Time: {abs(((100 * 200 * (3600 / mc1_rate)) - ((800 * (3600 / mc2_rate)))) + 1) - (abs((optimal_batch_size * optimal_mat_qty * (3600 / mc1_rate)) - (((optimal_mat_qty * 2000) * (3600 / mc2_rate)))) + 1)}')
    
    converter = CurrencyConverter()
    current_cost_usd = 2800
    current_cost_all = 20000 * 6
    optimal_cost_usd = optimal_tray * optimal_sets_qty * 6
    cost_delta_usd = current_cost_all - optimal_cost_usd
    current_cost_thb = converter.convert(current_cost_all, 'USD', 'THB')
    optimal_cost_thb = converter.convert(optimal_cost_usd, 'USD', 'THB')
    cost_delta_thb = current_cost_thb - optimal_cost_thb

    print(f'Current Cost per Mat/MC1 (6 USD/Mat): {current_cost_usd} USD or {current_cost_thb:.2f} Bath : All Mat use per MC1 ?? Box or ??? Mat cost = {20000 * 6} USD or {converter.convert(20000 * 6, "USD", "THB")} Bath') 
    print(f'Optimal Cost per Mat/MC1 (6 USD/Mat): {(optimal_cost_usd):.2f} USD or {optimal_cost_thb:.2f} Bath : Optimal All Process Lines Mat cost = {((optimal_box * optimal_mat_qty) * 6):.2f} USD or {converter.convert((optimal_box * optimal_mat_qty) * 1.4, "USD", "THB")} Bath')
    print(f'Delta Cost per Mat/MC1 USD: {cost_delta_usd:.2f} USD and THB: {cost_delta_thb:.2f} THB')

train_test_eva(X, y)
