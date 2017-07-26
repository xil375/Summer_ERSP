#Parsing_Dataset_3way.py
#coding: utf-8

#In[8]:

# This file contains a definition called parse_dataset_X that parses the files for the lean,obese,
# overweight samples.

# Method params: the metadata file name and the biom file name

# Return value: tuple (X, y) in which X is the 2D feature matrix and y is response vector
# the returned values corresponds to only lean/obese samples


# In[9]:

# Import statements
import biom
import sklearn
import pandas as pd

# In[11]:

# Function definition
#Note: Commented lines print out tables and lists

def parse_dataset_X(text_file_name, biom_file_name, isAmish):
    
    #read metadata file and load biom file
    m = pd.read_csv(text_file_name, sep="\t", index_col=0)
    b= biom.load_table(biom_file_name)
    
    # if file is amish, must convert sampleID from integer to string
    if (isAmish):
        m = pd.read_csv(text_file_name, sep="\t")
        m['help'] = m['#SampleID'].apply(str)
        m.set_index('help', inplace=True)
    
    i = m.index

    # create dataframe
    p = pd.DataFrame(b.matrix_data.T.todense().astype(int), index=b.ids(axis="sample"), columns=b.ids(axis="observation"))
    c = p.loc[i, :]
#   print(c)
    indices = pd.isnull(c).any(1).nonzero()[0]
    c[pd.isnull(c).any(axis=1)]
    c = c.dropna(how='any') 
    m = m.drop(m.index[indices])
    # return a tuple containing X and y
    return (c.fillna(0).as_matrix(), m.bmi_group_binned)

