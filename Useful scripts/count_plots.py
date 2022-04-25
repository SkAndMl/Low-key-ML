def count_plot(df,hue=None,cols=None):
  import seaborn as sns
  import pandas as pd
  import matplotlib.pyplot as plt
  sns.set_style("whitegrid")
  if cols==None:
    cols = []
    for i in df.columns:
      if(df[i].nunique()<10):
        cols.append(i)
  fig,axes = plt.subplots(nrows=len(cols),ncols=1,figsize=(10,5))
  for i,col in enumerate(cols):
    if hue:
      sns.countplot(data=df,x=col,ax=axes[i],hue=hue)
    else:
      sns.countplot(data=df,x=col,ax=axes[i])
  fig.tight_layout()
  plt.show()