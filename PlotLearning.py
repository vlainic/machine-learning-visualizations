import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure(figsize=(20, 10))
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        ### here is a bit tricky part
        ### the original code is made for the metrics accuracy
        ### I used different one, e.g. binary_accuracy, categorical_accuracy, mae, mse
        ### than you need to have following things instead of 'val_acc':
        ### 'val_binary_accuracy', 'val_categorical_accuracy', 'val_mean_absolute_error', 'val_mean_squared_error'
        self.acc.append(np.sqrt(logs.get('acc')))
        self.val_acc.append(np.sqrt(logs.get('val_acc')))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
        
        clear_output(wait=True)        
        plt.ticklabel_format(useOffset=False, style='plain') ### I am annoyed with "offset-tick" plotting that is common in Python
        
#         ax1.set_yscale('log') ### original
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.tick_params(labelsize=10) ### I added control of the fontsize
        ax1.legend(fontsize=9)
        
#         ax2.set_yscale('log') ### if you like, choose the metrics to be in log-scale too :)
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.tick_params(labelsize=10)  
        ax2.legend(fontsize=9)
        
        plt.show();
