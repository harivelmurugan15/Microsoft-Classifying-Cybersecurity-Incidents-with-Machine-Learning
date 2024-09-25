import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\ASUS\Downloads\GUIDE_Test.csv~\GUIDE_Test.csv")

# handeling null values

df.drop(['Timestamp','LastVerdict','SuspicionLevel','AntispamDirection','Roles','ResourceType','ThreatFamily','EmailClusterId','ActionGranular','ActionGrouped','MitreTechniques'],axis=1,inplace=True)

model = LabelEncoder()

for col in df.columns:
    df[col] = model.fit_transform(df[col])

X = df.drop(['IncidentGrade'],axis=1)
y = df['IncidentGrade']

df.to_csv("test_encoded.csv")

rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X,y = rus.fit_resample(X,y)

x_train,x_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=43)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf.fit(x_train, y_train)

feature_importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Sort by importance, descending
top_10_features = importance_df.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()

# Display top features
print("Top 10 Important Features:\n", top_10_features)

# Select only the top 10 important features from the training and validation sets
X_train_top_10 = x_train[top_10_features]
X_val_top_10 = x_val[top_10_features]

# Initialize a new Random Forest Classifier
rf_top_10 = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the top 10 features
rf_top_10.fit(X_train_top_10, y_train)

# Make predictions on the validation set using the top 10 features
y_pred_top_10 = rf_top_10.predict(X_val_top_10)

# Calculate the accuracy score
accuracy_top_10 = accuracy_score(y_val, y_pred_top_10)

# Print the accuracy
print(f"Validation Accuracy using top 10 features: {accuracy_top_10:.4f}")
print(f"f1_score : {f1_score(y_val, y_pred_top_10,average='macro')}")
print(f"precision_score : {precision_score(y_val, y_pred_top_10,average='macro')}")
print(f"recall_score : {recall_score(y_val, y_pred_top_10,average='macro')}")