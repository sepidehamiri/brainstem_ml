import warnings
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Dataset
KidneyDataset = pd.read_excel("Final_data.xlsx")
KidneyDataset = KidneyDataset.fillna(method='ffill')

X = KidneyDataset[['Age', 'Sex', 'TotalDose', 'AverageDoseOfBrainStem',
                      'FractionDose', 'Mean1', 'Minimum1', 'Maximum1',
                      'VoxelNum1', 'VolumeNum1', 'VoxelNum2', 'VolumeNum2', 'Mean2',
                      'Minimum2', 'Maximum2', 'VoxelVolume', 'Maximum3DDiameter', 'MeshVolume',
                      'MajorAxisLength', 'Sphericity', 'LeastAxisLength', 'Elongation',
                      'SurfaceVolumeRatio', 'Maximum2DDiameterSlice', 'Flatness', 'SurfaceArea',
                      'MinorAxisLength', 'Maximum2DDiameterColumn', 'Maximum2DDiameterRow',
                      'GrayLevelVariance1', 'HighGrayLevelEmphasis', 'DependenceEntropy',
                      'DependenceNonUniformity', 'GrayLevelNonUniformity', 'SmallDependenceEmphasis',
                      'SmallDependenceHighGrayLevelEmphasis', 'DependenceNonUniformityNormalized',
                      'LargeDependenceEmphasis', 'LargeDependenceLowGrayLevelEmphasis',
                      'DependenceVariance', 'LargeDependenceHighGrayLevelEmphasis',
                      'SmallDependenceLowGrayLevelEmphasis', 'LowGrayLevelEmphasis', 'JointAverage',
                      'SumAverage', 'JointEntropy', 'ClusterShade', 'MaximumProbability', 'Idmn',
                      'JointEnergy', 'Contrast1', 'DifferenceEntropy', 'InverseVariance',
                      'DifferenceVariance', 'Idn', 'Idm', 'Correlation', 'Autocorrelation',
                      'SumEntropy', 'MCC', 'SumSquares', 'ClusterProminence', 'Imc2', 'Imc1',
                      'DifferenceAverage', 'Id', 'ClusterTendency', 'InterquartileRange', 'Skewness',
                      'Uniformity', 'Median', 'Energy', 'RobustMeanAbsoluteDeviation',
                      'MeanAbsoluteDeviation', 'TotalEnergy', 'Maximum3', 'RootMeanSquared',
                      '90Percentile', 'Minimum3', 'Entropy', 'Range', 'Variance', '10Percentile',
                      'Kurtosis', 'Mean3', 'ShortRunLowGrayLevelEmphasis', 'GrayLevelVariance2',
                      'LowGrayLevelRunEmphasis', 'GrayLevelNonUniformityNormalized', 'RunVariance',
                      'GrayLevelNonUniformity', 'LongRunEmphasis', 'ShortRunHighGrayLevelEmphasis',
                      'RunLengthNonUniformity', 'ShortRunEmphasis', 'LongRunHighGrayLevelEmphasis',
                      'RunPercentage', 'LongRunLowGrayLevelEmphasis', 'RunEntropy',
                      'HighGrayLevelRunEmphasis', 'RunLengthNonUniformityNormalized',
                      'GrayLevelVariance3', 'ZoneVariance', 'GrayLevelNonUniformityNormalized',
                      'SizeZoneNonUniformityNormalized', 'SizeZoneNonUniformity',
                      'GrayLevelNonUniformity', 'LargeAreaEmphasis', 'SmallAreaHighGrayLevelEmphasis',
                      'ZonePercentage', 'LargeAreaLowGrayLevelEmphasis', 'LargeAreaHighGrayLevelEmphasis',
                      'HighGrayLevelZoneEmphasis', 'SmallAreaEmphasis', 'LowGrayLevelZoneEmphasis',
                      'ZoneEntropy', 'SmallAreaLowGrayLevelEmphasis', 'Coarseness', 'Complexity',
                      'Strength', 'Contrast2', 'Busyness']]
y = KidneyDataset['BrainStemDamage']

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
knn = neighbors.KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print('accuracy of knn classifier is:', accuracy)
scores = cross_val_score(knn, X, y, cv=10)
print(scores.mean())
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
total = sum(sum(cm))

# Calculate evaluation criteria
accuracy = (cm[0, 0] + cm[1, 1]) / total
print('accuracy:', accuracy)
sensivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('sensivity:', sensivity)
specifity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('specifity:', specifity)
cv_scores = cross_val_score(knn, X, y)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, knn.predict_proba(X)[:, 1])
print('auc:')
print(auc(false_positive_rate, true_positive_rate))
false_positive_rate, true_positive_rate, _ = roc_curve(y, knn.predict_proba(X)[:, 1])
