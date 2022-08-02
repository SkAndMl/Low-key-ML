import pandas as pd
from sklearn import preprocessing
class DataPreparation():
    
    def __init__(self,target=None,cat_cols=None,num_cols=None):
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
    
    def prepare_data(self,data,scale=True,one_hot=True,encode=False,drop_first=True):
        
        new_data = data.copy()
        self.drop_first = drop_first
        self.scale = scale
        self.one_hot = one_hot
        
        if one_hot==True and encode==True:
            raise ValueError("'one_hot' and 'encode' both are true. Only one of them can be true")
        
        if isinstance(self.cat_cols,str):
            self.cat_cols = (self.cat_cols,)
        
        if isinstance(self.num_cols,str):
            self.num_cols = (self.num_cols,)
        
        if scale:
            self.scaler = preprocessing.StandardScaler()
            new_data[self.num_cols] = self.scaler.fit_transform(data[self.num_cols].values)
        
        if self.one_hot:
            if drop_first:
                self.new_cols = pd.get_dummies(data[self.cat_cols],drop_first=True,columns=self.cat_cols)
            else:
                self.new_cols = pd.get_dummies(data[self.cat_cols],columns=self.cat_cols)
        
            new_data[self.new_cols.columns] = self.new_cols
            new_data.drop(self.cat_cols,axis=1,inplace=True)
        
        return new_data
    
    def transform_new_data(self,data):
        new_data = data.copy()
        if self.scale:
            new_data[self.num_cols] = self.scaler.transform(data[self.num_cols].values)
        if self.one_hot:
            if self.drop_first:
                new_cols = pd.get_dummies(data[self.cat_cols],drop_first=True,columns=self.cat_cols)
            else:
                new_cols = pd.get_dummies(data[self.cat_cols],columns=self.cat_cols)
            new_data[new_cols.columns] = new_cols
            new_data.drop(self.cat_cols,axis=1,inplace=True)
        
            if len(new_cols.columns) != len(self.new_cols.columns):
                raise ValueError(f"Column dimension do not match while one-hot encoding is being performed\n{self.new_cols.columns}\
                                \n{new_cols.columns}")
            
        
        return new_data