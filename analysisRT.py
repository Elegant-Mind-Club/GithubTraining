import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Important Variables
folderPath = "/Users/Mingda/Desktop/analysis files/SampleData/vrScaleData"
timeDelay = 150
plotTitle = "VR 2D Face Scaling RT"
plotXLabel = "Eccentricity (°)"
plotYLabel = "Reaction time (ms)"
plotYLimits = [400, 800]

# z* value for confidence interval
zScore = 1.96
pValThreshold = 0.05
leftIVRows = [0, 1, 2]  # Adjusted for zero-indexing
rightIVRows = [2, 3]    # Adjusted for zero-indexing

# Data pre-processing
# Imports all files in the folder, folderPath, and adds to "data"
fileList = [f for f in os.listdir(folderPath) if f.endswith('.csv')]
data = []

for file in fileList:
    dataImport = pd.read_csv(os.path.join(folderPath, file))
    # sets the participant ID for each file (used in the legend during graphing) to the first column of "data"
    participantID = file.split("-")[0]
    data.append([participantID, file, dataImport])

# remove all incorrect trials and trials > 1.2s from "data", then subtracts the time delay
for entry in data:
    currentData = entry[2]
    # removes incorrect trials
    currentData = currentData[currentData['Correct'] == "True"]
    # removes trials > 1200 ms
    currentData = currentData[((currentData['ReactionTime'] - currentData['ObjShowTime']) < 1200)]
    # subtracts time delay from "data" reaction time
    currentData['ReactionTime'] = currentData['ReactionTime'] - timeDelay
    entry.append(currentData)

# go through each data file to remove outliers from the fourth column and create the fifth column in "data"
for entry in data:
    currentData = entry[3]
    independentVar = currentData['Distance']
    uniqueIVs = independentVar.unique()
    dependentVar = currentData['ReactionTime'] - currentData['ObjShowTime']

    dataSeparatedByIV = pd.DataFrame(columns=["IndependentVar", "Data", "Mean", "Std Dev", "Std Err"])
    for iv in uniqueIVs:
        rtForCondition = dependentVar[independentVar == iv]
        # removes outliers
        rtForCondition = rtForCondition[~((rtForCondition - rtForCondition.mean()).abs() > 1.5 * rtForCondition.std())]
        dataSeparatedByIV = dataSeparatedByIV.append({
            "IndependentVar": iv,
            "Data": rtForCondition.values,
            "Mean": rtForCondition.mean(),
            "Std Dev": rtForCondition.std(),
            "Std Err": rtForCondition.std() / np.sqrt(len(rtForCondition))
        }, ignore_index=True)

    entry[3] = currentData[currentData['ReactionTime'].isin(dataSeparatedByIV['Data'].explode())]
    entry.append(dataSeparatedByIV)

# Plotting
colors = sns.color_palette("husl", len(data))

# GRAPH 1
plt.figure(figsize=(10, 6))
ax = plt.gca()
sns.set(style="whitegrid")
ax.set_title(plotTitle)
ax.set_xlabel(plotXLabel)
ax.set_ylabel(plotYLabel)
ax.set_ylim(plotYLimits)

plotXVals = range(1, len(data[0][4]) + 1)
customXTickLabels = data[0][4]['IndependentVar']
plt.xticks(plotXVals, customXTickLabels)

r2Values = np.zeros((len(data), 2))

for i, entry in enumerate(data):
    currentData = entry[4]
    currentColor = colors[i]
    sns.scatterplot(x=plotXVals, y=currentData['Mean'], color=currentColor, s=100, label=entry[0])
    plt.errorbar(plotXVals, currentData['Mean'], yerr=currentData['Std Err'] * zScore, fmt='none', c=currentColor)

    leftYData = currentData.iloc[leftIVRows]['Mean']
    leftXData = np.array(leftIVRows) + 1
    fitLineLeftCoefs = np.polyfit(leftXData, leftYData, 1)
    fitLineLeft = np.polyval(fitLineLeftCoefs, leftXData)
    r2Values[i, 0] = stats.linregress(leftXData, leftYData).rvalue ** 2

    rightYData = currentData.iloc[rightIVRows]['Mean']
    rightXData = np.array(rightIVRows) + 1
    fitLineRightCoefs = np.polyfit(rightXData, rightYData, 1)
    fitLineRight = np.polyval(fitLineRightCoefs, rightXData)
    r2Values[i, 1] = stats.linregress(rightXData, rightYData).rvalue ** 2

    plt.plot(leftXData, fitLineLeft, color=currentColor)
    plt.plot(rightXData, fitLineRight, color=currentColor)

    print(f"ParticipantID: {entry[0]}; Left r^2: {r2Values[i, 0]}; Right r^2: {r2Values[i, 1]}")

plt.legend()
plt.show()
