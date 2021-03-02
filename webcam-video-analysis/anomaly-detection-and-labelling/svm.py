from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.svm import OneClassSVM

def get_weighted_avg(row):
    distribution = row[['yaw', 'roll', 'pitch']].values
    weights = [65 , 10 , 5]
    numerator = sum([abs(distribution[i])*weights[i] for i in range(len(distribution))])
    denominator = sum(weights)

    return round(numerator/denominator,4)

finalDf  = pd.read_csv('final-data.csv')

df_normal = finalDf.query('actual == 0')
df_abnormal = finalDf.query('actual == 1')


print("normal :", df_normal.shape[0])
print("abnormal :", df_abnormal.shape[0])

contamination =  df_abnormal.shape[0] /(df_normal.shape[0] + df_abnormal.shape[0])

df = pd.concat([df_normal, df_abnormal])

names = df['timestamp']
df.drop(columns=['timestamp'], inplace=True)
df['avg'] = df.apply(lambda row : get_weighted_avg(row), axis=1 )

# df_scaler = StandardScaler().fit(df[['yaw','roll','pitch']].to_numpy())
# newDf = df_scaler.transform(df[['yaw', 'pitch', 'roll']].to_numpy())
# newDf = df[['yaw', 'roll','pitch']].to_numpy()
# newDf = df[['avg']]
df_scaler = StandardScaler().fit(df[['avg']].to_numpy())
newDf = df_scaler.transform(df[['avg']].to_numpy())

rng = np.random.RandomState(39)
train_percentage = 0.7
train_samples = round(train_percentage*df.shape[0])


accuracies = []
false_neg_percentage = []


from sklearn.model_selection import KFold

kf = KFold(n_splits=4)
for train, test in kf.split(df_normal) :

    train =  df_normal.iloc[train]
    test =  df_normal.iloc[test]

    test = pd.concat([test, df_abnormal])

    # train = df.head(train_samples)
    # test = df.tail(df.shape[0] -train_samples)
    # fit the model
    model = OneClassSVM(kernel = 'rbf', gamma = 0.0001, nu = contamination).fit(train[['yaw','roll','pitch']])

    # test = clf.predict(newDf)
    score = model.predict(test[['yaw', 'roll','pitch']])

    a = plt.figure(1).gca()
    a.hist(score)
    a.set_title('Histogram of anomaly score')
    a.set_xlabel("Anomaly Score")
    a.set_ylabel("Number of instances")
    test['scores'] = score
    # test['names'] = names
    test['pred'] = test.apply(lambda row: 1 if row['scores'] == -1 else 0, axis=1)

    outliers = test.query('pred == 1')
    normal = test.query('pred == 0')

    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(normal['pitch'],
                    normal['roll'], normal['yaw'], color="#00FF00", label="normal points")
    threedee.scatter(outliers['pitch'],
                    outliers['roll'], outliers['yaw'], color="#FF0000", label="anomalies")

    threedee.legend(loc="center left")
    threedee.set_xlabel('Pitch')
    threedee.set_ylabel('Roll')
    threedee.set_zlabel('Yaw')
    threedee.set_title(
        'Predicted normal points and anomalies with One Class SVM')

    # print(silhouette_score(df['actual'].actuals, df['pred'].actuals))

    # print('accuracy')
    # print(accuracy_score(test['actual'], test['pred']))
    accuracies.append(accuracy_score(test['actual'], test['pred']))

    # print('confusion matrix')
    # print('TP FP')
    # print('FN TN')
    tn, fp, fn, tp = confusion_matrix(test['actual'], test['pred']).ravel()
    conf_mat = confusion_matrix(test['actual'], test['pred'])

    # print(conf_mat)
    # print(test.query('pred == 1 and actual == 0').shape[0])

    # print(fp)
    fn = conf_mat[0][1]
    total = sum(conf_mat[0])
    false_neg_percentage.append(fn/total)

# plt.show()

# print("Identified false Negatives")
# print(df.query('pred == 1 and actual == 0'))

# print("Identified false Negatives")

print(accuracies)
print(false_neg_percentage)


from statistics import mean

print(mean(accuracies))
print(mean(false_neg_percentage))
