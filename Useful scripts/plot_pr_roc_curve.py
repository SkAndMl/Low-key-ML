def plot_pr_roc_curve(model,X,y,test=True,test_size=0.2,random_state=42):
  """
  This function is used to plot both the precision-recall curve and roc-curve for both training and test
  data after splitting the data.
  model -> base estimator
  X -> feature set
  y -> target set
  test -> if you have not trained the model yet, test is usedto take care of that. Default value is true
  """
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import precision_recall_curve,roc_curve
  import seaborn as sns
  import matplotlib.pyplot as plt
  if test:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    model = model.fit(X_train,y_train)
    pred_t,pred_tr = model.predict(X_test),model.predict(X_train)
    p_t,r_t,thresh = precision_recall_curve(y_test,pred_t)
    tpr_t,fpr_t,thresh = roc_curve(y_test,pred_t)
    p_tr,r_tr,thresh = precision_recall_curve(y_train,pred_tr)
    tpr_tr,fpr_tr,thresh = roc_curve(y_train,pred_tr)
    sns.set_style("whitegrid")
    fig,ax = plt.subplots(ncols=2)
    ax[0].plot(p_t,r_t,label="test")
    ax[0].set_title("Precision Recall Curve",size=15)
    ax[0].set_xlabel("Precision")
    ax[0].set_ylabel("Recall")
    ax[0].plot(p_tr,r_tr,label="train")
    ax[0].legend()
    ax[1].plot(tpr_t,fpr_t,label="test")
    ax[1].set_xlabel("TPR")
    ax[1].set_ylabel("FPR")
    ax[1].set_title("Roc Curve",size=15)
    ax[1].plot(tpr_tr,fpr_tr,label="train")
    ax[1].legend()
    fig.tight_layout()
  else:
    pred = model.predict(X)
    p,r,thr = precision_recall_curve(y,pred)
    tpr,fpr,thr = roc_curve(y,pred)
    fig,ax = plt.subplots(ncols=2)
    ax[0].plot(p,r)
    ax[0].set_title("Precision Recall Curve",size=15)
    ax[0].set_xlabel("Precision")
    ax[0].set_ylabel("Recall")
    ax[1].plot(tpr,fpr)
    ax[1].set_title("Roc Curve",size=15)
    ax[1].set_xlabel("TPR")
    ax[1].set_ylabel("FPR")
  plt.show()