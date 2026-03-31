import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load dataset
ufo = pd.read_csv("ufo_sightings.csv")
pd.set_option('display.max_columns', None)
print(ufo)

# states with more than 2000 sightings
sightings_per_state = ufo['Location.State'].value_counts()
high_sighting_states = sightings_per_state[sightings_per_state > 2000].index

# pie chart for high sighting states
plt.figure()
plt.title("US States with Over 2000 UFO Sightings")
plt.pie(sightings_per_state[sightings_per_state > 2000], labels=high_sighting_states)
plt.show()

# most common UFO shapes
shape_counts = ufo['Data.Shape'].value_counts()
plt.figure()
sns.barplot(x=shape_counts.index, y=shape_counts.values)
plt.title('Most Commonly Reported UFO Shapes')
plt.xlabel('UFO Shape')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# compare top shapes by state
top_shapes = ufo.groupby(['Location.State', 'Data.Shape']).size().unstack()[['light', 'triangle', 'disk']]
top_shapes.plot(kind='bar', stacked=True, title='Top UFO Shapes by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.show()

# filter disk shapes
disk_reports = ufo[ufo['Data.Shape'] == 'disk']

# before and after 1996
before_movie = disk_reports[
    (disk_reports['Dates.Sighted.Year'] >= 1993) &
    (disk_reports['Dates.Sighted.Year'] < 1996)
]

after_movie = disk_reports[
    (disk_reports['Dates.Sighted.Year'] > 1996) &
    (disk_reports['Dates.Sighted.Year'] <= 2000)
]

# print counts
print("Disk shapes reported:")
print("Before movie=", len(before_movie))
print("After movie=", len(after_movie))

total_before = ufo[
    (ufo['Dates.Sighted.Year'] >= 1993) &
    (ufo['Dates.Sighted.Year'] < 1996)
]

total_after = ufo[
    (ufo['Dates.Sighted.Year'] > 1996) &
    (ufo['Dates.Sighted.Year'] <= 2000)
]

print("Total sighting in 3 years:")
print("Before movie=", len(total_before))
print("After movie=", len(total_after))

# probability comparison
print("Probability of disk UFO reported")
print("Before movie=", (len(before_movie) / len(total_before)) * 100, "%")
print("After movie=", (len(after_movie) / len(total_after)) * 100, "%")

# line plot before vs after
before_counts = before_movie['Dates.Sighted.Year'].value_counts().sort_index()
after_counts = after_movie['Dates.Sighted.Year'].value_counts().sort_index()

plt.figure()
sns.lineplot(x=before_counts.index, y=before_counts, marker='o', label='Before (1993-1996)')
sns.lineplot(x=after_counts.index, y=after_counts, marker='o', label='After (1997-2000)')
plt.axvline(x=1996, linestyle='--', label='Independence Day (1996)')
plt.title('Disk-Shaped UFO Reports Before/After Movie Release')
plt.xlabel('Year')
plt.ylabel('Number of Reports')
plt.show()

# median duration over time
duration_trend = ufo.groupby('Dates.Sighted.Year')['Data.Encounter duration'].median()
duration_trend.plot()
plt.title('Median UFO Sighting Duration Over Time')
plt.xlabel('Year')
plt.ylabel('Duration (seconds)')
plt.show()

# shape vs duration (under 2 hours)
duration_df = ufo[['Data.Shape', 'Data.Encounter duration']].dropna()
duration_df = duration_df[duration_df['Data.Encounter duration'] <= 120]

shape_medians = duration_df.groupby('Data.Shape')['Data.Encounter duration'].median()
shape_counts = duration_df['Data.Shape'].value_counts()

common_shapes = shape_counts[shape_counts >= 30].index
shape_medians = shape_medians[common_shapes].sort_values()

plt.figure()
sns.barplot(x=shape_medians.index, y=shape_medians.values)
plt.xticks(rotation=45)
plt.title('Median Encounter Duration by UFO Shape (≤2 hours)')
plt.ylabel('Median Duration (minutes)')
plt.xlabel('UFO Shape')
plt.tight_layout()
plt.show()

# fix column names
ufo.columns = ufo.columns.str.strip()

# keep only 5 shapes
ufo = ufo[
    (ufo["Data.Shape"] == "light") |
    (ufo["Data.Shape"] == "triangle") |
    (ufo["Data.Shape"] == "circle") |
    (ufo["Data.Shape"] == "changing") |
    (ufo["Data.Shape"] == "disk")
]

# features and target
x = ufo[
    [
        "Data.Encounter duration",
        "Location.Coordinates.Latitude",
        "Location.Coordinates.Longitude",
        "Dates.Sighted.Year",
        "Dates.Sighted.Month",
        "Date.Sighted.Day",
        "Dates.Sighted.Hour",
        "Dates.Sighted.Minute"
    ]
]

y = ufo["Data.Shape"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=16
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred, normalize='true')

sns.heatmap(
    cnf_matrix,
    annot=True,
    fmt='.2f',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.title("Normalized Confusion Matrix - UFO Shape Prediction")
plt.tight_layout()
plt.show()

# feature importance
features = [
    "Data.Encounter duration",
    "Location.Coordinates.Latitude",
    "Location.Coordinates.Longitude",
    "Dates.Sighted.Year",
    "Dates.Sighted.Month",
    "Date.Sighted.Day",
    "Dates.Sighted.Hour",
    "Dates.Sighted.Minute"
]

plt.figure()
plt.xticks(rotation=45)
plt.title("Feature Importance - UFO Shape Prediction")
plt.bar(features, model.feature_importances_)
plt.tight_layout()
plt.show()
