import pandas as pd
import numpy as np


combined = pd.read_csv('combined.csv')
# combined = combined.set_index('timestamp')
# print(combined)
# exit(0)
# select only the correctly predicted values
combined = combined.query('face_count == 1')

totalFiles = combined['timestamp']

aslamNormal = []
irfanNormal = []

aslamAbnormal = pd.read_csv('aslam.txt', header=None,names=['filename'])

irfanAbnormal = pd.read_csv('irfan.txt', header=None,names=['filename'])

for filename in totalFiles:
    if filename not in aslamAbnormal.filename.tolist():
        aslamNormal.append(filename)

for filename in totalFiles:
    if filename not in irfanAbnormal.filename.tolist():
        irfanNormal.append(filename)


commonAbnormal = pd.merge(aslamAbnormal, irfanAbnormal, how='inner', left_on='filename', right_on='filename')

commonNormal = pd.merge(pd.DataFrame(aslamNormal,columns =['filename']), pd.DataFrame(irfanNormal,columns =['filename']), how='inner', left_on='filename', right_on='filename')



differences = []

for filename in irfanAbnormal.filename.tolist():
    if filename not in commonAbnormal.filename.tolist():
        differences.append(filename)

for filename in aslamAbnormal.filename.tolist():
    if filename not in commonAbnormal.filename.tolist():
        differences.append(filename)

# print(len(differences))


for filename in irfanNormal:
    if filename not in commonNormal.filename.tolist():
        differences.append(filename)

for filename in aslamNormal:
    if filename not in commonNormal.filename.tolist():
        differences.append(filename)


# print(len(differences))



multipleFaces = pd.read_csv('multiplefaces.csv')

multipleFaces = multipleFaces['timestamp'].tolist()

for filename in differences:
    if filename in multipleFaces:
        differences.remove(filename)



myset = set(differences)
differences = list(myset)


print("Differences : % 4d, Abnormal : % 4d, Normal : % 4d" %(len(differences), len(commonAbnormal),len(commonNormal) ))

for filename in differences:
    row = combined.loc[combined['timestamp'] == filename]
    yaw = row['yaw'].values[0]
    pitch = row['pitch'].values[0]
    roll = row['roll'].values[0]
    if abs(yaw) > 25 or abs(pitch) > 5 or abs(roll) > 10 :
        commonAbnormal = commonAbnormal.append({'filename' : filename}, ignore_index=True)
    else:
        commonNormal = commonNormal.append({'filename' : filename}, ignore_index=True)


print("Total : % 4d, Abnormal : % 4d, Normal : % 4d" %(len(combined), len(commonAbnormal),len(commonNormal) ))


combined['actual'] = combined.apply(lambda row: 0 if row['timestamp'] in commonNormal.filename.tolist() else 1, axis=1)


combined.to_csv('final-data.csv', index=False)
test =  combined.drop(columns=['timestamp'])

print(test)
test.to_csv('preview-final-data.csv', index=False)


# for filename in multipleFaces['timestamp']:
#     copyfile('/home/irfan/Desktop/Uni/FYP/FSA-Net/demo/combined/'+filename,'/home/irfan/workspace/isolation-forest/multiple-faces-fsa/'+filename)
