#!/usr/bin/env python
# coding: utf-8

# # Correlation analysis of crimes in NSW

# Python code to analyse relationships among crimes in NSW based on data from the NSW Bureau of Crime Statistics and Research.

# # Loading libraries

# Let's load relevant Python libraries.

# In[11]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from IPython.display import display, Markdown
from itertools import cycle
import time
import warnings
warnings.filterwarnings("ignore")


# # Loading data

# The original data acquired from https://www.bocsar.nsw.gov.au/Pages/bocsar_datasets/Datasets-.aspxNSW. It contains monthly data on all criminal incidents recorded by police from 1995 to Mar 2020. It was processed and cleaned in another notebook. Now we are going to use the processed file.

# In[2]:


data = pd.read_csv('crimes_nsw.csv', index_col=0, parse_dates=True)


# In[3]:


data.head()


# # Exploratory data analysis

# Let's analyze all the 62 types of crimes. 

# In[4]:


np.random.seed(4)
colors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(data.columns), replace=False)
cycler = cycle('bgrcmk')


# In[5]:


for i in range(len(data.columns)):
    data[data.columns[i]].plot(kind = 'line', color =colors[i],linewidth=2,alpha = 1,grid = True,linestyle = '-')
plt.legend(loc='upper right')     
plt.legend(bbox_to_anchor=(1.05, 1))   
plt.xlabel('')              
plt.ylabel('Number of cases')
plt.title('Monthly plot of various crimes in NSW')            
plt.show()


# Let's look at individual plots.

# In[6]:


np.random.seed(4)
colors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(data.columns), replace=False)
for i in range(len(data.columns)):
    data[data.columns[i]].plot(kind = 'line', color=colors[i],linewidth=2,alpha = 1,grid = True,linestyle = '-')
    plt.legend(loc='upper right')     
    plt.legend(bbox_to_anchor=(1.05, 1))   
    plt.xlabel('')              
    plt.ylabel('Number of cases')
    plt.title('')            
    plt.show()


# # Set up for corelation analysis

# Let's compute pairwise Pearson correlation of columns

# In[7]:


corr = data.corr()


# In[8]:


corr.head()


# Let's plot a heatmap of the correlation matrix where highly correlated pairs are colored in pink.

# In[9]:


mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5)


# Since the correlation table is too big, let's loop through column by column. Let's set the threshold as 0.8 and find strongly positively and negatively correlated columns.

# In[10]:


threshold = 0.8
for column in corr.columns:
    targetcolumn = corr[column]
    filteredcolumn = targetcolumn[((targetcolumn>threshold) & (targetcolumn<1))|
                                  ((targetcolumn<-threshold) & (targetcolumn>-1))]
    
    display(Markdown("# Crimes correlated with '" + column + "'"))
    if len(filteredcolumn)==0:
         display(Markdown('None'))
    else:
        print(filteredcolumn)
        print('')
        print('')
        np.random.seed(53)
        colors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(filteredcolumn)+1, replace=False)
        data[column].plot(kind = 'line', color = colors[0],label = column,linewidth=2,alpha = 1,grid = True,linestyle = '-')
        i=1
        for index, value in list(filteredcolumn.items()):            
            data[index].plot(kind = 'line', color = colors[i],label = index,linewidth=2,alpha = 1,grid = True,linestyle = '-')
            i=i+1
        plt.legend(loc='upper right')     
        plt.legend(bbox_to_anchor=(1.05, 1))   
        plt.xlabel('')              
        plt.ylabel('Number of cases')
        plt.title('Crimes correlated with '+ column.lower())            
        plt.show()
                
        for index, value in list(filteredcolumn.items()): 
            print('')    
            display(Markdown("### Relationship between '"+ column + "' and '" + index + "'" ))
            if value > 0:
                if value > threshold * 1.125:
                    print('There is a strong positive correlation with a coefficient of ' + str(value) + ".")
                else:
                    print('There is a somewhat a weak positive correlation with a coefficient of ' + str(value) + ".")
            else:
                if value > threshold * -1.125:                    
                    print('There is a strong negative correlation with a coefficient of ' + str(-value) + '.')
                else:
                    print('There is a somewhat a weak negative correlation with a coefficient of ' + str(-value) + ".")
                
            sns.jointplot(x=column, y=index, data=data, kind="reg")  
            plt.show()             
    print('')   
    print('')
    print('')
            
   
   

