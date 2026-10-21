import streamlit as st
import joblib
import pandas as pd


@st.cache_resource
def load_model():
    return joblib.load('models/student_performance_model.pkl')
model = load_model()


st.title('Student Performance Prediction')

st.write(
    "Enter the student's information to predict whether they are "
    'likely to pass or be at risk.'
)

st.info(
    "This is an educational prediction tool, not a definitive "
    "assessment of a student's ability."
)

studytime = st.selectbox(
    'Weekly Study Time',
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: 'Less than 2 hours',
        2: '2 to 5 hours',
        3: '5 to 10 hours',
        4: 'More than 10 hours'
    }[x],
    index=1
)


failures = st.number_input(
    'Past Class Failures',
    min_value=0,
    max_value=3,
    value=0
)

absences = st.number_input(
    'School Absences',
    min_value=0,
    max_value=100,
    value=4
)

schoolsup = st.selectbox(
    'Extra School Support',
    ['no', 'yes']
)

famsup = st.selectbox(
    'Extra family support',
    ['no', 'yes']
)

paid = st.selectbox(
    'Paid extra class',
    ['no', 'yes']
)

higher = st.selectbox(
    'Wants higher education',
    ['yes', 'no']
)

Medu = st.selectbox(
    "Mother's Education",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: 'None',
        1: 'Primary education',
        2: '5th to 9th grade',
        3: 'Secondary education',
        4: 'Higher education'
    }[x],
    index=2
)

Fedu = st.selectbox(
    "Father's Education",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {
        0: 'None',
        1: 'Primary education',
        2: '5th to 9th grade',
        3: 'Secondary education',
        4: 'Higher education'
    }[x],
    index=2
)

traveltime = st.selectbox(
    'Travel Time',
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: 'Less than 15 min',
        2: '15 to 30 min',
        3: '30 to 60 min',
        4: 'More than 1 hours'
    }[x],
    index=1
)



if st.button('Predict Performance'):
    student = {
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'schoolsup': schoolsup,
        'famsup': famsup,
        'paid': paid,
        'higher': higher,
        'Medu': Medu,
        'Fedu': Fedu,
        'traveltime': traveltime
    }

    student_df=pd.DataFrame([student])
    result=model.predict(student_df)[0]

    st.subheader('Prediction')

    if result == 'Pass':
        st.success('PASS ✅')
    else:
        st.error('AT RISK ⚠️')

    p = model.predict_proba(student_df)[0][list(model.classes_).index('At Risk')]
    st.write(f'Estimated probability of being at risk: **{p:.0%}**')