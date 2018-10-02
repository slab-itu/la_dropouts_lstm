This anonymous repository is for the dataset and code used for the paper entitled "Leveraging Deep Learning to Predict Withdrawal of Students in Virtual Learning Environment" submitted to LAK 2019.

Find the code below followed by the sample data, note that the anonymous repository comes with limited space, so complete data will be provided with actual physical link in camera-ready if the paper is accepted – thank you.


=====================================CODE=======================================


import pandas as pd
import numpy as np
from keras.layers import LSTM
import keras
np.random.seed()


df1=pd.read_csv('38Weeks-WithPass-Data.csv', low_memory=False)

df1=df1.sort_values(by=['id1','week_id'])


###### factorizing the final result....pass=0, fail=1
d=[ 'final_result']

for val in d:
    labels,levels = pd.factorize(df1[val])
    df1[val] = labels


df1.head()





############################# Same length window ########################
def make_frames1(df):
    
    
    trys = []
    big_flat=[]
    for x in range(0,25): 
        
        trys.append(list(df.iloc[0:x+1,1:21].values[x])) 
        flat_list=[]
        for sublist in trys:         
            for item in sublist:
                flat_list.append(item)
        
        big_flat.append(flat_list) 

    
    #padding
    for i in range(0,25):
        op=len(big_flat[i])
        for w in range(0,(500-op)):
            big_flat[i].append(-1)
    
    data_a=pd.DataFrame({"bigflat":big_flat}) 
    return data_a



def make_labels(df):
    labels=[]
    for x in range(0,25):
        labels.append(list(df.iloc[0:x+1,1:2].values[x]))
    return labels


df_label=pd.DataFrame({})
for num in df1['id1'].unique():    
    t=make_labels(df1[df1['id1']==num])#make frame fr each unique id
    
    
    print("DF for one id:",t)
    df_label=df_label.append(t) # the total df that has all the row for each id
                        # 0-37 for 1 id, 38-75 for 2nd id...(75+38=113)....
    
print("DF-labels: ", df_label)
    
print(type(df_label))

df1=df1.drop(['final_result'], axis=1)


#fr each unique id, it will create dataframes, 
#### appending 0-37 rows for each unique id 
dfnew=pd.DataFrame({})
for num in df1['id1'].unique():    
    t=make_frames1(df1[df1['id1']==num])#make frame fr each unique id
    #t=t.sort_values(by=['id1'])
    
    print("DF for one id:",t)
    #print("t",t)
    dfnew=dfnew.append(t) # the total df that has all the row for each id
                        # 0-37 for 1 id, 38-75 for 2nd id...(75+38=113)....
    
print("DF-Final: ", dfnew)
    
print(dfnew.shape)
print(type(dfnew))


df_col = pd.DataFrame(dfnew['bigflat'].values.tolist())



X_train= df_col.iloc[0:180000,] 
y_train= df_label.iloc[0:180000,]
X_test= df_col.iloc[180000:,]
y_test=df_label.iloc[180000:,]




##### In[111]:


print(type(X_train))
print(type(y_train))
print(type(X_test))
print(type(y_test))
print("xtrain",X_train.shape)
print("ytrain",y_train.shape)
print("xtest",X_test.shape)
print(y_test.shape)
print(X_train.shape[1])


X_train=X_train.values.reshape((X_train.shape[0], X_train.shape[1] ))
y_train=y_train.values.reshape((y_train.shape[0],1 ))
X_test=X_test.values.reshape((X_test.shape[0],X_test.shape[1]))
y_test=y_test.values.reshape((y_test.shape[0],1))


print(X_train.shape)
print("type xtrain", type(X_train))
print(y_train.shape)
print("type ytrain", type(y_train))
print(X_test.shape)
print(y_test.shape)




print(X_train.shape[0])
print(X_train.shape[1])
print(X_test.shape[0])



X_train=X_train.reshape((X_train.shape[0],25,20))
X_test=X_test.reshape((X_test.shape[0],25,20))



print(X_train.shape[0])
print(X_train.shape[1])
print("xtrain.shape[2]",X_train.shape[2])
print(X_train.shape)
print(X_test.shape[0])
print(X_test.shape[1])
print(X_test.shape[2])
print(X_test.shape)

print(y_train.shape[1])




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



=====================================Data=====================================
week_id	final_result	DataPlus	DualPane	ExternalQuiz	Folder	Forumng	Glossary	Homepage	HtmlActivity	OuCollaborate	Oucontent	OuElluminate	OuWiki	Page	Questionnaire	Quiz	RepeatActivity	Resource	SharedSubpage	SubPage	Url	id1
1	Pass	0	0	0	0	109	0	39	0	0	27	0	1	3	0	2	0	2	0	22	8	0
2	Pass	0	0	0	0	20	0	12	0	0	5	0	0	0	0	0	0	2	0	8	7	0
3	Pass	0	0	0	0	10	0	4	0	0	3	0	0	0	0	0	0	0	0	3	0	0
4	Pass	0	0	0	0	4	0	8	0	0	54	0	12	0	0	0	0	0	0	5	3	0
5	Pass	0	0	0	0	4	0	3	0	0	47	0	0	0	0	0	0	0	0	3	0	0
6	Pass	0	0	0	0	11	0	17	0	0	70	0	0	2	0	0	0	6	0	28	1	0
7	Pass	0	0	0	0	42	0	29	0	1	97	0	2	0	0	0	0	5	0	23	0	0
8	Pass	0	0	0	0	17	0	26	0	0	99	0	8	0	3	0	0	0	0	18	3	0
9	Pass	0	6	0	0	4	0	12	0	0	90	0	0	0	7	0	0	2	0	3	0	0
10	Pass	0	0	0	0	29	0	37	0	0	122	0	2	0	0	0	0	2	0	29	1	0
11	Pass	0	0	0	0	15	0	19	0	0	32	0	0	0	0	0	0	0	0	17	0	0
12	Pass	0	0	0	0	18	0	6	0	0	19	0	0	0	0	0	0	0	0	2	0	0
13	Pass	0	5	0	0	11	0	25	0	0	128	0	0	0	0	3	0	1	0	23	2	0
14	Pass	0	0	0	0	12	0	15	0	0	69	0	0	1	0	27	0	2	0	26	1	0
15	Pass	0	0	0	0	11	0	42	0	0	357	0	0	0	0	0	0	10	0	43	0	0
16	Pass	0	0	0	0	6	0	29	0	0	217	0	0	0	0	0	0	6	0	35	3	0
17	Pass	0	0	0	0	10	0	28	0	0	195	0	5	0	0	0	0	0	0	17	0	0
18	Pass	0	0	0	0	10	0	16	0	0	88	0	8	0	0	0	0	0	0	7	1	0
19	Pass	0	0	0	0	15	0	19	0	0	90	0	1	1	0	0	0	2	0	8	2	0
20	Pass	0	0	0	0	39	0	27	0	0	74	0	1	0	9	0	0	0	0	11	1	0
21	Pass	0	0	0	0	44	0	62	0	0	147	0	0	0	0	234	0	1	0	28	3	0
22	Pass	5	0	0	0	34	0	35	0	2	145	0	7	1	6	0	0	7	0	24	5	0
23	Pass	0	0	0	0	2	0	14	0	0	80	0	0	0	0	0	0	1	0	6	4	0
24	Pass	0	0	0	0	0	0	6	0	0	50	0	0	0	0	12	0	1	0	7	0	0
25	Pass	0	1	0	0	8	0	18	0	0	135	0	0	4	3	0	0	4	0	15	4	0
26	Pass	0	0	0	0	6	0	58	0	0	177	0	0	0	0	159	0	0	0	37	0	0
27	Pass	18	2	0	0	1	0	19	0	0	107	0	0	0	3	99	0	1	0	18	0	0
28	Pass	18	2	0	0	1	0	19	0	0	107	0	0	0	3	99	0	1	0	18	0	0
29	Pass	0	2	0	0	15	0	17	0	0	66	0	0	0	3	28	0	3	0	13	0	0
30	Pass	0	0	0	1	8	0	26	0	0	202	0	0	0	0	89	0	0	0	17	0	0
31	Pass	0	0	0	0	11	0	24	0	0	159	0	0	0	0	0	0	0	0	10	3	0
32	Pass	0	0	0	3	6	0	21	0	0	65	0	0	0	0	0	0	4	0	23	0	0
33	Pass	0	0	0	0	3	0	8	0	0	0	0	9	2	0	0	0	1	0	1	0	0
34	Pass	0	2	0	1	2	0	23	0	0	153	0	0	0	0	0	0	1	0	10	2	0
35	Pass	0	0	0	0	2	0	2	0	0	0	0	0	0	0	0	0	0	0	0	0	0
36	Pass	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
37	Pass	0	0	0	0	2	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
38	Pass	0	0	0	0	4	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
1	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
2	Withdrawn	0	0	0	0	5	0	3	0	0	0	0	0	0	0	0	0	0	0	1	0	1
3	Withdrawn	0	0	0	0	1	0	15	0	0	0	0	0	0	0	0	0	0	0	0	0	1
4	Withdrawn	0	0	0	0	5	0	2	0	0	0	0	0	0	0	0	0	1	0	1	1	1
5	Withdrawn	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	1
6	Withdrawn	0	0	0	0	0	0	4	0	0	0	0	0	0	0	1	0	0	0	1	0	1
7	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
8	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
9	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
10	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
11	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
12	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
13	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
14	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
15	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
16	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
17	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
18	Withdrawn	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
