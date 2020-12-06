import warnings
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import model_selection
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, SelectFromModel, SelectPercentile

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load Dataset
KidneyDataset = pd.read_excel("100_final_data.xlsx")
KidneyDataset = KidneyDataset.fillna(method='ffill')
X = KidneyDataset[['Age', 'Sex', 'TotalDose', 'AverageDoseOfBrainStem',
                      'FractionDose', 'CochleaDose', 'EarDamage', 'Mean1', 'Minimum1', 'Maximum1',
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
X = SelectKBest(k=20).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=4)
clf = RandomForestClassifier(n_estimators=8, max_depth=3, max_features=6, n_jobs=-1,
                             min_samples_leaf=4, random_state=30, min_samples_split=2)
clf.fit(X_train, y_train)
print('accuracy of RF for BrainStemDataset on training set is: {:.2f}'.format(clf.score(X, y)))
print('accuracy of RF for BrainStemDataset on testing set is: {:.2f}'.format(clf.score(X_test, y_test)))
y_pred = clf.predict(X_test)

# Calculate evaluation criteria
cm = confusion_matrix(y_test, y_pred)
print(cm)
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print('accuracy:', accuracy)
sensivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('sensivity:', sensivity)
specifity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('specifity:', specifity)
cv_scores = cross_val_score(clf, X, y)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
accuracy=(cross_val_score(clf,X,y,scoring='accuracy',cv=5).mean()*100)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])
print('auc:')
print(auc(false_positive_rate, true_positive_rate))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])

# Find and sort important features
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances.to_string())
with open('RF_feature_importances.txt', 'a') as the_file:
    the_file.write(str(feature_importances.to_string()))
