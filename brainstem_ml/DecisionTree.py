import warnings
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2, SelectFromModel
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

warnings.simplefilter(action='ignore', category=FutureWarning)

# Read Dataset
BrainStemDataset = pd.read_excel("Final_data.xlsx")
BrainStemDataset = BrainStemDataset.fillna(method='ffill')

X = BrainStemDataset[['Age', 'Sex', 'TotalDose', 'AverageDoseOfBrainStem',
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
y = BrainStemDataset['BrainStemDamage']

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
clf = DecisionTreeClassifier(max_depth=3, max_features=4, min_samples_leaf=4, presort=True,
                             random_state=30, max_leaf_nodes=5).fit(X, y)
clf.fit(X_train, y_train)

# Calculate evaluation criteria
print('accuracy of RF for BrainStemDataset on training set is: {:.2f}'.format(clf.score(X, y)))
print('accuracy of RF for BrainStemDataset on testing set is: {:.2f}'.format(clf.score(X_test, y_test)))
y_pred = clf.predict(X_test)
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
# print('cross validation scores:', cv_scores)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))

seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
accuracy = (cross_val_score(clf, X, y, scoring='accuracy', cv=5).mean() * 100)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])
print('auc:')
print(auc(false_positive_rate, true_positive_rate))
false_positive_rate, true_positive_rate, _ = roc_curve(y, clf.predict_proba(X)[:, 1])

# Find and sort important features
feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
