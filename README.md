This anonymous repository is for the dataset and code used for the paper entitled "Leveraging Deep Learning to Predict Withdrawal of Students in Virtual Learning Environment" submitted to LAK 2019.

Find the sample data below followed by the code, note that the anonymous repository comes with limited space, so complete data will be provided with actual physical link in camera-ready if the paper is accepted – thank you.


=====================================Data=====================================


=====================================CODE=======================================





====================================LSTM Code==================================


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import  Dropout
import keras



from keras.layers import LSTM, Masking
import tensorflow as tf
import keras.callbacks
import sys
import os
from keras import backend as K

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))



###### record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])
        print("check point")



#LSTM model
sequence_length=10
nb_features = X_train.shape[2] #20
nb_out = y_train.shape[1] #1

model = Sequential()

model.add(Masking(mask_value=-1, input_shape=(sequence_length, nb_features)))

model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=300,
         return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(
          units=200,
          return_sequences=True))
model.add(Dropout(0.5))


model.add(LSTM(
          units=100,
          return_sequences=False))
model.add(Dropout(0.5))


if checkpoint:
    model.load_weights(checkpoint)

file_name = os.path.basename(sys.argv[0]).split('.')[0]
check_cb = keras.callbacks.ModelCheckpoint('LSTM-try/'+ file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

history = LossHistory()
#keras.optimizers.RMSprop(lr=0.001)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',precision,recall, f1])

pandas.DataFrame(model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=400,
          epochs=60, shuffle=True, callbacks=[ check_cb, history]).history).to_csv("10(new)-week-WithPass-tanh.csv")


model.save('10(new)-weeksModel-WithPass.h5')

