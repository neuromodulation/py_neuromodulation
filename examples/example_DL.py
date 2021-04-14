import sys
import pandas as pd
import numpy as np
sys.path.append(r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\github files\py_neuromodulation\pyneuromodulation")

from sklearn.model_selection import KFold, TimeSeriesSplit

from dl_archs import tabnet_regression
import training_utils as tu
import tensorflow as tf


# Settings

gradient_clipping = True
GRADIENT_THRESH = 2000.0



# importing the data
path = r"D:\Jupyter notebooks\Interventional Cognitive Neuromodulation\github files\py_neuromodulation\pyneuromodulation\tests\data\derivatives\sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg\sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg_FEATURES.csv"

df = pd.read_csv(path)

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
epochs = 100
epoch_print = 1

tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    batch_percentage = 10
    test_batch_num = int(np.floor(X_test.shape[0]//batch_percentage))
    batch_size = int(np.floor(X_train.shape[0]//batch_percentage))
    model = tabnet_regression(num_features=X.shape[1]
                              , output_dim=1
                              , feature_dim=X.shape[1]
                              , num_decision_steps=6
                              , relaxation_factor=1.5
                              , batch_momentum=0.7
                              , epsilon=0.00001
                              , BATCH_SIZE=batch_size
                              , virtual_batch_size=batch_size)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)


    tr_dataset = tf.data.Dataset.from_tensor_slices((X_train.iloc[0:batch_size*batch_percentage], y_train.iloc[0:batch_size*batch_percentage]) )
    # te_dataset = tf.data.Dataset.from_tensor_slices((X_test.iloc[0:batch_size*test_batch_num], y_test.iloc[0:batch_size*test_batch_num]) )

    train_loss_results = []

    for epoch in range(epochs):

        epoch_loss_avg = tf.keras.metrics.Mean()

        for feat, targ in tr_dataset.take(batch_size):
            loss_value, grads = tu.grad(model, feat, targ)

            if gradient_clipping:
                capped_gvs = [(tf.clip_by_value(grad, -GRADIENT_THRESH,
                                                GRADIENT_THRESH), var) for grad, var in grads]

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)


        train_loss_results.append(epoch_loss_avg.result())

        if epoch % epoch_print == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result() ) )





