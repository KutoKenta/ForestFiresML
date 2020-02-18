import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools	


plt.style.use('ggplot')

import statsmodels.api as sm
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from scipy.stats import zscore
from statsmodels.stats.stattools import durbin_watson
import tensorflow as tf
import logging
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

logger = tf.get_logger()
logger.setLevel(logging.INFO)

target = 'area'
Features          = ['FFMC','DMC','DC','ISI','temp','RH','wind']
TrainingSetLabels = ['FFMC','DMC','DC','ISI','temp','RH','wind','area']


# path = 'forestfires.csv'
path = "forestfires.csv"
df = pd.read_csv(path)
training_set = pd.read_csv("train.csv")
prediction_set = pd.read_csv("test.csv")
df.shape
df.dtypes
df.describe().T
df.isna().sum().sum()
plt.rcParams["figure.figsize"] = 9,5

# Outlier points
y_outliers = df[abs(zscore(df[target])) >= 3 ]
y_outliers

dfa = df.drop(columns=target)
cat_columns = dfa.select_dtypes(include='object').columns.tolist()
num_columns = dfa.select_dtypes(exclude='object').columns.tolist()

cat_columns,num_columns

#print(df['area'].describe(),'\n')
#print(y_outliers)

# a categorical variable based on forest fire area damage
# No damage, low, moderate, high, very high
def area_cat(area):
    if area == 0.0:
        return "No damage"
    elif area <= 1:
        return "low"
    elif area <= 25:
        return "moderate"
    elif area <= 100:
        return "high"
    else:
        return "very high"

df['damage_category'] = df['area'].apply(area_cat)
df.head()

cat_columns

selected_features = df.drop(columns=['damage_category','day','month']).columns
selected_features

out_columns = ['area','FFMC','ISI','rain']



#Preparing Data
df = pd.get_dummies(df,columns=['day','month'],drop_first=True)



#FFMC and rain are still having high skew and kurtosis values, 
# since we will be using Linear regression model we cannot operate with such high values
# so for FFMC we can remove the outliers in them using z-score method
mask = df.loc[:,['FFMC']].apply(zscore).abs() < 3

# Since most of the values in rain are 0.0, we can convert it as a categorical column
df['rain'] = df['rain'].apply(lambda x: int(x > 0.0))

df = df[mask.values]
df.shape
out_columns.remove('rain')
df[out_columns] = np.log1p(df[out_columns])
df[out_columns].skew()

# we will use this dataframe for building our ML model
df_ml = df.drop(columns=['damage_category']).copy()
TrainingSet = df[TrainingSetLabels]


feature_cols = [tf.feature_column.numeric_column(k) for k in Features]			

#print(training_set)
#TrainingSet.to_csv ('test1.csv', index = None, header=True)
####################################################################################################################
################################################Keras###############################################################


train_dataset = training_set.sample(frac=0.8,random_state=0)
print(train_dataset)

test_dataset = training_set.drop(train_dataset.index)
print(test_dataset)


train_stats = train_dataset.describe()
train_stats.pop("area")
train_stats = train_stats.transpose()
train_stats

train_labels = train_dataset.pop('area')
test_labels = test_dataset.pop('area')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)



def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

test_predictions = model.predict(normed_test_data).flatten()
print(test_predictions)

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.savefig('foo.png')

model.save('saved_model/my_model') 
