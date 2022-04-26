def tt_with_sc(df,test_size=0.2,random_state=42,include_cols=None):
  import numpy as np
  from sklearn.model_selection import train_test_split
  import pandas as pd
  from sklearn.preprocessing import StandardScaler

  np.random.seed(42)

  X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],
                                                   test_size=test_size,random_state=random_state)
  sc = StandardScaler()
  if not include_cols:
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
  else:
    sc_train_cols = X_train[include_cols].copy()
    sc_test_cols = X_test[include_cols].copy()
    sc_train_cols = sc.fit_transform(sc_train_cols)
    sc_test_cols = sc.transform(sc_test_cols)
    X_train[include_cols] = sc_train_cols
    X_test[include_cols] = sc_test_cols
    X_train,X_test = X_train.values,X_test.values
  return X_train,X_test,y_train,y_test