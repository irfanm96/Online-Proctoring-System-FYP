import pandas as pd
import numpy as np

# input = 'combined'
# input = 'mtcnn-res'
input = 'lbp-res'
confidence = 0.95

rawDataFrame = pd.read_csv(input+ '.csv')


stat = rawDataFrame.groupby(['face_count']).agg(['count'])

print(stat)

multipleFaces = rawDataFrame.query('face_count != 1')
# singleFaces = rawDataFrame.query('face_count == 1')


# print('Stats with face count in FSA-net algorithm\n')

# print('Accuracy with confidence level 0.5')
# print(singleFaces.shape[0]*100/ rawDataFrame.shape[0])

# print()

# print("False positives interms of number of faces")
# print(multipleFaces.shape[0]*100/ rawDataFrame.shape[0])


# print(rawDataFrame.query('face_count == 2').shape[0])
# multipleFaces.to_csv( filename +'-multiplefaces.csv', index= False)


from shutil import copy2

for filename in multipleFaces['timestamp']:
    copy2('/home/irfan/Desktop/Uni/FYP/FSA-Net/demo/original-combined/'+filename,'/home/irfan/workspace/isolation-forest/multiple-faces-fsa/'+ input + '/')
