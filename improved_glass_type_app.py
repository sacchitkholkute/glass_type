# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe):
  glass_type=model.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
  glass_type=glass_type[0]
  if glass_type==1:
    return "building windows float processed".upper()
  elif glass_type==2:
    return "building windows non float processed".upper()
  elif glass_type==3:
    return "vehicle windows float processed".upper()
  elif glass_type==4:
    return "vehicle windows non float processed".upper()
  elif glass_type==5:
    return "containers".upper()
  elif glass_type==6:
    return "tableware".upper()
  else:
    return "headlamp".upper()
st.sidebar.title("Exploratory Data Analysis")
st.title("Glass Type Predictor")
if st.sidebar.checkbox("Show raw data"):
  st.subheader("Full Data Set")
  st.dataframe(glass_df)
plot_list=st.sidebar.multiselect('Select the Chart/Plots',['Correlation Heatmap','Line Chart','Area Chart','Count Plot','Pie Chart','Box Plot'])
if 'Line Chart' in plot_list:
  st.subheader('LINE CHART')
  st.line_chart(glass_df)
if 'Area Chart' in plot_list:
  st.subheader('AREA CHART')
  st.area_chart(glass_df)
  st.set_option('deprecation.showPyplotGlobalUse',False)
if 'Count Plot' in plot_list:
  st.subheader('COUNT PLOT')
  sns.countplot(x='GlassType',data=glass_df)
  st.pyplot()
if 'Pie Chart' in plot_list:
  st.subheader('PIE CHART')
  plt.figure(figsize=(10,10))
  plt.pie(glass_df['GlassType'].value_counts(),autopct='%1.1f%%',shadow=True,explode=np.linspace(0,0.12,6))
  st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse',False)
if 'Count Plot' in plot_list:
  st.subheader('COUNT PLOT')
  sns.countplot(x='GlassType',data=glass_df)
  st.pyplot()
if 'Box Plot' in plot_list:
  st.subheader('Box Plot')
  column=st.sidebar.selectbox('Select the column for Box Plot',['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','GlassType'])
  sns.boxplot(glass_df[column])
  st.pyplot()
st.sidebar.subheader('Select Your Values')
RI=st.sidebar.slider('input RI',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
Na=st.sidebar.slider('input NA',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
Mg=st.sidebar.slider('input Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
Al=st.sidebar.slider('input Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
Si=st.sidebar.slider('input Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
K=st.sidebar.slider('K',float(glass_df['K'].min()),float(glass_df['K'].max()))
Ca=st.sidebar.slider('input Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
Ba=st.sidebar.slider('input Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
Fe=st.sidebar.slider('input Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machines', 'Logistic Regression', 'Random Forest Classifier'))
from sklearn.metrics import plot_confusion_matrix
if classifier=='Support Vector Machines':
  st.sidebar.subheader('Model Hyperparameters:') 
  c_value=st.sidebar.number_input('C (Error Rate)',1,10,step=1)
  kernel_input=st.sidebar.radio('Kernel',('linear','rbf','poly'))
  gamma_input=st.sidebar.number_input('Gamma',1,10,step=1)
  if st.sidebar.button('Classify'):
    svc_model=SVC(C=c_value,kernel=kernel_input,gamma=gamma_input)
    svc_model.fit(X_train,y_train)
    y_pred=svc_model.predict(X_test)
    score=svc_model.score(X_test,y_test)
    glass_type=prediction(svc_model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
    st.write('Glass Type',glass_type)
    st.write('Accuracy Score',score)
    plot_confusion_matrix(svc_model,X_test,y_test)
    st.pyplot()
if classifier=='Random Forest Classifier':
  st.sidebar.subheader("Model Hyperparameters:")
  n_estimators_input=st.sidebar.number_input('N_estimators',100,5000,step=10)
  max_depth=st.sidebar.number_input('Max_depth',1,100,step=1)
  if st.sidebar.button('Classify'):
    rf_clf=RandomForestClassifier(n_estimators = n_estimators_input, max_depth=max_depth_input, n_jobs = -1)
    rf_clf.fit(X_train,y_train)
    rf_score=rf_clf.score(X_test,y_test)
    glass_type=prediction(rf_clf,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
    st.write('Glass Type',glass_type)
    st.write('Accuracy Score',rf_score)
    plot_confusion_matrix(rf_clf,X_test,y_test)
    st.pyplot()
if classifier=='Logistic Regression':
  st.sidebar.subheader('Model Hyperparameters:')
  c_value=st.sidebar.number_input('C (Error rate)',1,100,step=1)
  max_iter_input=st.sidebar.number_input('Max Iterations',10,1000,step=10)
  if st.sidebar.button('Classify'):
    log_reg=LogisticRegression(C=c_value,max_iter=max_iter_input)
    log_reg.fit(X_train,y_train)
    score=log_reg.score(X_test,y_test)
    glass_type=prediction(log_reg,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
    st.write('Glass Type Predicted:',glass_type)
    st.write('Accuracy Score:',round(score,2))
    plot_confusion_matrix(log_reg,X_test,y_test)
    st.pyplot()