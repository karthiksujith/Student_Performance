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

print('\nshape:\n',df.shape)

print('\ncolumns:\n',df.columns)

df.info()

print('\nnull values:\n',df.isnull().sum())

print('\nduplicated rows:\n',df.duplicated().sum())

print('\nG3 value counts:\n',df['G3'].value_counts().sort_index())

print('\nstatistics:\n',df.describe())


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

print('\nperformance distribution:\n',df['performance'].value_counts())

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

print('\nFeatures(X):\n',X.columns.to_list())
print('\nno of features:\n',X.shape[1])
print('\nTarget(y):\n',y.head())


numerical_features = X.select_dtypes(
    include=['int64','float64']
).columns

categorical_features = X.select_dtypes(
    include=['object','category','str']
).columns

print('\nNumerical features:\n',list(numerical_features))

print('\nCategorical features:\n',list(categorical_features))


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print('\nTraining data shape:\n',X_train.shape)
print('\nTest data shape:\n',X_test.shape)
print('\nTraining target:\n',y_train.value_counts())
print('\nTest target:\n',y_test.value_counts())


preprocessor=ColumnTransformer(
    transformers=[
        ('numerical',StandardScaler(),numerical_features),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ]
)
print('\npreprocessor created successfully')

model=Pipeline([
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model.fit(X_train,y_train)
print('\nModel trained successfully')

pred=model.predict(X_test)
acc=accuracy_score(y_test,pred)
recall=recall_score(y_test,pred,pos_label='At Risk')
f1=f1_score(y_test,pred,pos_label='At Risk')
report=classification_report(y_test,pred)
cm=confusion_matrix(y_test,pred,labels=['At Risk','Pass'])

print('\nPredictions:\n',pred[:10])
print('\nAccuracy:\n',acc)
print('\nrecall:\n',recall)
print('\nf1:\n',f1)
print('\nclassification_report:\n',report)
print('\ncm:\n',cm)

joblib.dump(model, 'models/student_performance_model.pkl')
print('\nModel saved successfully!')