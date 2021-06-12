#!/usr/bin/env python
# coding: utf-8

# # Forecasting_1

# ### TASK: FORECASTING

# Forecasting the  Airlines Passengers data set

# IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
import seaborn as sns


# IMPORTING DATA

# In[2]:


airlines=pd.read_excel('C:/Users/vinay/Downloads/Airlines+Data.xlsx')


# In[3]:


airlines.head()


# In[38]:


airlines.info()


# In[39]:


airlines.describe()


# ## Histogram and Density Plots

# In[69]:


# create a histogram plot
airlines.hist()
pyplot.show()


# In[70]:


airlines.Passengers.plot(kind='kde')


# ## LinePlot

# In[45]:


series1 = pd.read_excel('C:\ExcelrPy\Assignment-18\Airlines+Data.xlsx', header=0, index_col=0)
series1.plot()
pyplot.show()


# In[47]:


series1


# ### Lag plot

# In[50]:


# create a scatter plot
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
lag_plot(series1)
pyplot.show()


# ### ACF PLot

# In[52]:


# create an autocorrelation plot
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series1,lags=30)
pyplot.show()


# # SquareTransformation

# In[53]:


from pandas import read_csv
from pandas import DataFrame
from numpy import sqrt
from matplotlib import pyplot


# # Upsampling the data to each and every day

# In[56]:


upsampled = series1.resample('D').mean()
print(upsampled.head(32))


# In[57]:


##### interpolate the missing value
interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
pyplot.show()


# # Before Transformation 

# In[60]:


# line plot
pyplot.subplot(211)
pyplot.plot(interpolated)
# histogram
pyplot.subplot(212)
pyplot.hist(interpolated)
pyplot.show()


# In[61]:


interpolated


# #### Square Root Transform

# In[62]:


dataframe = DataFrame(interpolated)
dataframe.columns = ['Passengers']
dataframe['Passengers'] = sqrt(dataframe['Passengers'])


# In[63]:


# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Passengers'])
pyplot.show()


# In[64]:


interpolated


# #### Log Transform

# In[65]:


from numpy import log
dataframe = DataFrame(interpolated)
dataframe.columns = ['Passengers']
dataframe['Passengers'] = log(dataframe['Passengers'])

# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Passengers'])
pyplot.show()


# In[66]:


interpolated


# In[68]:


interpolated.info()


# # Plotting Heatmap

# In[71]:


airlines


# In[72]:


airlines["Date"]=pd.to_datetime(airlines.Month,format="%b-%y")
airlines["Months"]=airlines.Date.dt.strftime("%b")
airlines["Year"]=airlines.Date.dt.strftime("%Y")


# In[73]:


# Heatmap
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=airlines,values="Passengers",index="Year",columns="Month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# In[75]:



sns.boxplot(x="Months",y="Passengers",data=airlines)
sns.boxplot(x="Year",y="Passengers",data=airlines)


# In[76]:



Month_Dummies = pd.DataFrame(pd.get_dummies(airlines['Months']))
airline1 = pd.concat([airlines,Month_Dummies],axis = 1)


# In[77]:


airline1["t"] = np.arange(1,97)
airline1["t_squared"] = airline1["t"]*airline1["t"]
airline1["Log_Passengers"] = np.log(airline1["Passengers"])


# In[129]:


airline1


# In[79]:



plt.figure(figsize=(12,3))
sns.lineplot(x="Year",y="Passengers",data=airlines)


# In[80]:



Train = airline1.head(80)
Test = airline1.tail(16)


# In[ ]:





# In[81]:


# Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[82]:



# Exponential Model
Exp = smf.ols('Log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[83]:



# Quadratic Model
Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[84]:



# Additive seasonality
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[94]:


# Additive Seasonality quadrative

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad=pd.Series(add_sea_Quad.predict(Test[['t','t_squared','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[85]:


#Multiplicative Seasonality

Mul_sea = smf.ols('Log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[86]:


#Multiplicative addditive seasonality

Mul_Add_sea = smf.ols('Log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# TESTING

# In[95]:




data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


# In[ ]:





# ### PREDICT WITH NEW DATA

# In[133]:


t = np.arange(97,108)
t


# In[134]:


t_squared=np.array(t*t)
t_squared


# In[148]:


Month = pd.DataFrame({'Month':['2003-01-01','2003-02-01','2003-03-01','2003-04-01','2003-05-01','2003-06-01','2003-07-01',
                               '2003-08-01','2003-09-01','2003-10-01','2003-10-01']})


# In[149]:


df={'t':t,'t_squared':t_squared}
df=pd.DataFrame(df)


# In[150]:


newdata = pd.concat([Month,df],axis=1)
newdata


# ### Build the model on entire dataset

# In[151]:


model_full = smf.ols('Passengers~t',data=airline1).fit()
pred_new  = pd.Series(model_full.predict(newdata))
pred_new


# In[152]:


newdata["forecasted_passengers"]=pd.Series(pred_new)


# In[153]:


newdata


# OBSERVATION:
#     
# Multiplicative Additive Seasonality gives the best prediction of least RMSE of 9.42    

# In[ ]:




