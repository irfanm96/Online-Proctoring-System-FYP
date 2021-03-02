from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

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
# fit the model
clf = IsolationForest(random_state=rng,behaviour="new", contamination="auto")
clf.fit(newDf)
# test = clf.predict(newDf)
score = clf.decision_function(newDf)

a = plt.figure(1).gca()
a.hist(score)
a.set_title('Histogram of anomaly score')
a.set_xlabel("Anomaly Score")
a.set_ylabel("Number of instances")
df['scores'] = score
df['names'] = names
df['pred'] = df.apply(lambda row: 1 if row['scores'] < -0.12 else 0, axis=1)

outliers = df.query('pred == 1')
normal = df.query('pred == 0')

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
    'Predicted normal points and anomalies with Isolation Forest')

# print(silhouette_score(df['actual'].actuals, df['pred'].actuals))

print('accuracy')
print(accuracy_score(df['actual'], df['pred']))

print('confusion matrix')
print('TP FP')
print('FN TN')
print(confusion_matrix(df['actual'], df['pred']))
# plt.show()

# print("Identified false Negatives")
# print(df.query('pred == 1 and actual == 0'))

# print("Identified false Negatives")
# print(df.query('pred == 0 and actual == 1'))
