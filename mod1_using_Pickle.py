from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np                                   # optimize arrays
import pandas as pd                                  # data analytics
import matplotlib.pyplot as plt                      # data visualization
import tensorflow as tf                              # needed to create a linear regression model algo
import tensorflow.compat.v2.feature_column as fc     # 
# from IPython.display import clear_output             # to enable clearing the output


# "gender","race/ethnicity","parental level of education",
# "lunch","test preparation course","math score","reading score",
# "writing score"


# "gender", "lunch", "math score","reading score","writing score"

data = pd.read_csv("Data/StudentsPerformance.csv")
# print(data.head())
# #    gender race/ethnicity  ... reading score writing score
# # 0  female        group B  ...            72            74
# # 1  female        group C  ...            90            88
# # 2  female        group B  ...            95            93
# # 3    male        group A  ...            57            44
# # 4    male        group C  ...            78            75

# # [5 rows x 8 columns]

# print(type(data))
# # <class 'pandas.core.frame.DataFrame'>

# shuffle the data
shuffled_data = data.reindex(np.random.permutation(data.index))
# print(shuffled_data.head())
# print(shuffled_data.shape)
# # (1000, 8)
# print(shuffled_data.shape[0])
# # 1000
# print(len(shuffled_data))
# # 1000


# break data into sets 60% training, 20% validation, 20% test
train_data = shuffled_data.sample(frac=0.6)
remaining_data = shuffled_data.drop(train_data.index)
# print(f"train_data: {len(train_data)}")                    # 600
# print(f"remaining_data: {len(remaining_data)}")            # 400
validation_data = remaining_data.sample(frac=.5)
test_data = remaining_data.drop(validation_data.index)
# print(f"validation_data: {len(validation_data)}")            # 200
# print(f"test_data: {len(test_data)}")                        # 200



# # store "math score" column from each df as a variable
y_train = train_data.pop("math_score")
y_test = test_data.pop("math_score")
# print(y_train)
# # 873    90
# # 638    86
# # 725    81
# # 320    67
# # 207    81
# #        ..
# # 354    59
# # 379    66
# # 550    79
# # 330    71
# # 941    78
# # Name: math score, Length: 600, dtype: int64

# y_train_first_five = y_train.head()
# print(y_train_first_five)

# # inspect values at specific locations (in this case row 0)
# print(train_data.loc[0])
# # gender                                    female
# # race/ethnicity                           group B
# # parental level of education    bachelor's degree
# # lunch                                   standard
# # test preparation course                     none
# # reading score                                 72
# # writing score                                 74
# # Name: 0, dtype: object


# # Feature Columns
CATEGORICAL_COLUMNS = ["gender","race_ethnicity",
                        "parental_level_of_education",
                        "lunch","test_preparation_course"
                        ]
NUMERIC_COLUMNS = ["reading_score","writing_score"]

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # gets a list of all unique values from given feature column
    vocabulary = train_data[feature_name].unique()  
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

# for feature_name in NUMERIC_COLUMNS:
#     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# # print(feature_columns)
# # [VocabularyListCategoricalColumn(key='sex', vocabulary_list=('male', 'female'),
# # dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategorical
# # Column(key='n_siblings_spouses', vocabulary_list=(1, 0, 3, 4, 2, 5, 8), dtype=tf
# # .int64, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(ke
# # y='parch', vocabulary_list=(0, 1, 2, 5, 3, 4), dtype=tf.int64, default_value=-1,
# #  num_oov_buckets=0), VocabularyListCategoricalColumn(key='class', vocabulary_lis
# # t=('Third', 'First', 'Second'), dtype=tf.string, default_value=-1, num_oov_bucke
# # ts=0), VocabularyListCategoricalColumn(key='deck', vocabulary_list=('unknown', '
# # C', 'G', 'A', 'B', 'D', 'F', 'E'), dtype=tf.string, default_value=-1, num_oov_bu
# # ckets=0), VocabularyListCategoricalColumn(key='embark_town', vocabulary_list=('S
# # outhampton', 'Cherbourg', 'Queenstown', 'unknown'), dtype=tf.string, default_val
# # ue=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='alone', vocabula
# # ry_list=('n', 'y'), dtype=tf.string, default_value=-1, num_oov_buckets=0), Numer
# # icColumn(key='age', shape=(1,), default_value=None, dtype=tf.float32, normalizer
# # _fn=None), NumericColumn(key='fare', shape=(1,), default_value=None, dtype=tf.fl
# # oat32, normalizer_fn=None)]

# # print(train_data["sex"].unique())
# # # ['male' 'female']

# # print(train_data["embark_town"].unique())
# # # ['Southampton' 'Cherbourg' 'Queenstown' 'unknown']


# # especially for larger data sets feeding the data into the program in batches can 
# # make it easier for the computer to handle

# # an epoch is one stream of the entire dataset fed in a different order, ie if there are 10 epochs the model
# # will see the data 10 times

# # overfitting - if the computer is fed the same data too many times it can make a model too
# # specific to the data set, the computer essentially memorizes the dataset and develops a 
# # model that predicts poorly when fed other sets. This is why it is better to start with
# # a lower number of epochs and adjust up from there if needed

# # Because we feed the data in batches and multiple times we need to create an input 
# # function. This will define how the dataset will be converted into batches at each
# # epoch, how the data will be broken into epochs
# def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
#     def input_function():            #inner function, this will be returned
#         # create tf.data.Dataset object with data and it's label
#         ds = tf. data.Dataset.from_tensor_slices((dict(data_df), label_df))
#         if shuffle:
#             # randomize the order of the data
#             ds = ds.shuffle(1000)
#         # split the dataset into batches and repeat the process for number of epochs
#         ds = ds.batch(batch_size).repeat(num_epochs)
#         return ds
#     return input_function

# # prepare the datasets for the model
# train_input_fn = make_input_fn(train_data, y_train)
# eval_input_fn = make_input_fn(df_eval, y_eval, num_epochs=1, shuffle=False)


# # creating the model
# linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# # training the model
# # train
# linear_est.train(train_input_fn)
# # get model metrics/stats by testing on testing data
# result = linear_est.evaluate(eval_input_fn)

# # clear the console output
# clear_output()
# # the result is a dict of stats about the model
# print(result["accuracy"])
# # 0.74242425
# # this means at first pass the model is about 74% accurate
# # this may change each run as the computer may interpret 
# # the data differently each time it is shuffled

# # # what is stored in result
# # print(result)
# # # {'accuracy': 0.7348485, 'accuracy_baseline': 0.625, 'auc': 0.82133454, 'auc_precision_recall': 0.7615293, 'average_loss': 0.58852434, 'lab
# # # el/mean': 0.375, 'loss': 0.5903315, 'precision': 0.610687, 'prediction/mean': 0.52226925, 'recall': 0.8080808, 'global_step': 200}

# # if we want to actually check the predictions from the model
# # made into a list here so we can loop through it
# result_dict = list(linear_est.predict(eval_input_fn))
# # print(result_dict)
# # # see /Resources/result_output.txt


# # print(result_dict[0])
# # # {'logits': array([-2.1910503], dtype=float32), 'logistic': array([0.10055706], dtype=float32), 'probabilities': array([0.899443  , 0.10055
# # # 706], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_c
# # # lasses': array([b'0', b'1'], dtype=object)}


# # # check the survival probability of the first entry
# # print(result_dict[0]["probabilities"])
# # # [0.9670563  0.03294368]
# # # so about 97% chance of death, 3% chance of survival

# # # let's take a look at the stats of this person
# # print(df_eval.loc[0])
# # # sex                          male
# # # age                          35.0
# # # n_siblings_spouses              0
# # # parch                           0
# # # fare                         8.05
# # # class                       Third
# # # deck                      unknown
# # # embark_town           Southampton
# # # alone                           y
# # # Name: 0, dtype: object

# # # let's see if this person actually survived
# # print(y_eval.loc[0])
# # # 0
# # # 0 means did not survive (as mapped earlier 0 death, 1 survived)


# # # !figuring out conversion
# # print(result_dict[0]["probabilities"][1])
# # probs = pd.Series([pred["probabilities"][1]].astype(float) for pred in result_dict)
# # probs.astype(float)
# # print(probs)
# # # 0       [0.07950622]
# # # 1       [0.35783988]
# # # 2        [0.7451134]
# # # 3       [0.66329175]
# # # 4       [0.27935258]
# # #            ...
# # # 259      [0.8176828]
# # # 260    [0.083364256]
# # # 261      [0.5624116]
# # # 262     [0.19784631]
# # # 263     [0.41237444]
# # # Length: 264, dtype: object

# # print(probs)
# # print(type(probs[0]))
# # print(probs[0][0])
# # 0.07263894


# # # to display the predictions for survival of every person in the dataset
# # probs = pd.Series([pred["probabilities"][1]] for pred in result_dict)

# # probs_num = pd.Series([p[0] for p in probs])
# # # print(probs_num)
# # probs_num.plot(kind='hist', bins=20, title='predicted probabilities')
# # plt.show()



