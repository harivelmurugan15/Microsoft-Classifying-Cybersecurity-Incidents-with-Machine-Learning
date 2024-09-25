import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"C:\Users\ASUS\Downloads\new_train_sample.csv")

# handling null values

df.drop(['Unnamed: 0','Timestamp','LastVerdict','SuspicionLevel','AntispamDirection','Roles','ResourceType','ThreatFamily','EmailClusterId','ActionGranular','ActionGrouped','MitreTechniques'],axis=1,inplace=True)

# handel target column

df['IncidentGrade'] = df['IncidentGrade'].fillna(df['IncidentGrade'].mode()[0])

model = LabelEncoder()

for col in df.columns:
    df[col] = model.fit_transform(df[col])

df.to_csv('encoded_data.csv',index=False)