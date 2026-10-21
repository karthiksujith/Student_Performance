import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

df=pd.read_csv('data/student.csv',sep=';')

print(df.head())

print('shape:',df.shape)

print('columns:',df.columns)

df.info()

print('null values:',df.isnull().sum())

print('duplicated rows:',df.duplicated().sum())

print('G3 value counts:',df['G3'].value_counts().sort_index())

print('statistics:',df.describe())


plt.figure(figsize=(10,5))
plt.hist(df['G3'],bins=20,edgecolor='black')
plt.title('Distribution of Final Grade')
plt.xlabel('Final Grade ')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()


df['performance'] = df['G3'].apply(
    lambda grade: 'Pass' if grade >= 10 else 'At Risk'
)

print('performance distribution:',df['performance'].value_counts())

FEATURES=[
    'studytime',
    'failures',
    'absences',
    'schoolsup',
    'famsup',
    'paid',
    'higher',
    'Medu',
    'Fedu',
    'traveltime'
]

X=df[FEATURES]
y=df['performance']

print('Features(X):',X.columns.to_list())
print('no of features:',X.shape[1])
print('Target(y):',y.head())


numerical_features = X.select_dtypes(
    include=['int64','float64']
).columns

categorical_features = X.select_dtypes(
    include=['object','category','str']
).columns

print('Numerical features:',list(numerical_features))

print('Categorical features:',list(categorical_features))


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print('Training data shape:',X_train.shape)
print('Test data shape:',X_test.shape)
print('Training target:',y_train.value_counts())
print('Test target:',y_test.value_counts())


preprocessor=ColumnTransformer(
    transformers=[
        ('numerical',StandardScaler(),numerical_features),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ]
)
print('preprocessor created successfully')

model=Pipeline([
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model.fit(X_train,y_train)
print('Model trained successfully')

pred=model.predict(X_test)
acc=accuracy_score(y_test,pred)
recall=recall_score(y_test,pred,pos_label='At Risk')
f1=f1_score(y_test,pred,pos_label='At Risk')
report=classification_report(y_test,pred)
cm=confusion_matrix(y_test,pred,labels=['At Risk','Pass'])

print('Predictions:',pred[:10])
print('Accuracy:',acc)
print('recall:',recall)
print('f1:',f1)
print('classification_report:',report)
print('cm:\n',cm)

joblib.dump(model, 'models/student_performance_model.pkl')
print('Model saved successfully!')