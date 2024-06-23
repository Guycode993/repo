ECHO is on.
#hi
Hello!
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
df
y=df['logS']
y
X=df.drop('logS', axis=1) #axis=1 means columns

X
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,random_state=100)
X_train
X_test
y_train
y_test
#model building with linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
#applying the model to make the prediction
y_lr_train_pred=lr.predict(X_train)
y_lr_test_pred=lr.predict(X_test)
print(y_lr_train_pred)
print(y_lr_test_pred)
#evaluate model performance with comparing actual value and predicted value
from sklearn.metrics import mean_squared_error,r2_score
lr_train_mse=mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2=r2_score(y_train,y_lr_train_pred)
lr_test_mse=mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2=r2_score(y_test,y_lr_test_pred)

#y_train
#y_lr_train_pred
lr_train_mse
lr_train_r2
lr_test_mse
lr_test_r2
lr_results=pd.DataFrame(['Linear regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2])
lr_results
lr_results=pd.DataFrame(['Linear regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results
lr_results.columns=['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
lr_results
#Random forest test
#y variable is number or quantitative, so it is regressional model and if the y variable is categorical and it is classifier model

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(X_train, y_train)
y_rf_train_pred=rf.predict(X_train)
y_rf_test_pred=rf.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score
rf_train_mse=mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2=r2_score(y_train,y_rf_train_pred)
rf_test_mse=mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2=r2_score(y_test,y_rf_test_pred)
rf_results=pd.DataFrame(['Random Forest',rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2])
rf_results
rf_results=pd.DataFrame(['Random Forest',rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results
rf_results.columns=['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
rf_results
df_models=pd.concat([lr_results,rf_results],axis=0)
df_models
df_models.reset_index()
df_models.reset_index(drop=True)
