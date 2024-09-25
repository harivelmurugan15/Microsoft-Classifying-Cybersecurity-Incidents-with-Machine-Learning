from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd


top_10_features = ['OrgId', 'IncidentId', 'AlertId', 'DetectorId', 'AlertTitle', 'Id', 'Category', 'AccountUpn', 'EntityType', 'AccountObjectId']
df = pd.read_csv(r"encoded_data.csv")

X = df[top_10_features]
y = df['IncidentGrade']

rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X,y = rus.fit_resample(X,y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),GradientBoostingClassifier()]

for model in models:
    model.fit(x_train, y_train)

    # Make predictions
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    # Print the accuracy
    print(f"Validation Accuracy using top 10 features: {accuracy_score(train_pred,y_train)}")
    print(f"f1_score : {f1_score(train_pred,y_train,average='weighted')}")
    print(f"precision_score : {precision_score(train_pred,y_train,average='weighted')}")
    print(f"recall_score : {recall_score(train_pred,y_train,average='weighted')}")
    print('\n')
    print("Test_score")
    print(f"Validation Accuracy using top 10 features: {accuracy_score(test_pred,y_test)}")
    print(f"f1_score : {f1_score(test_pred,y_test,average='weighted')}")
    print(f"precision_score : {precision_score(test_pred,y_test,average='weighted')}")
    print(f"recall_score : {recall_score(test_pred,y_test,average='weighted')}")
    print('\n')