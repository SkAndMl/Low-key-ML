from plot_2 import plot_2
from train_test_split import train_test_split
def train_validation_curve_for_rf(X,y,val_size=0.3,rs=42,epochs=10,n_estimators=False,max_depth=True,
                                 max_depth_start=6,n_estimators_start=100,criterion='gini'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score
    import matplotlib.pyplot as plt
    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=val_size,random_state=rs)
    acc_tr,acc_val = [0.5],[0.5]
    pr_tr,pr_val = [],[]
    rec_tr,rec_val = [],[]
    if max_depth:
        for depth in range(max_depth_start,max_depth_start+epochs):
            rf = RandomForestClassifier(criterion=criterion,max_depth=depth,n_estimators=100)
            rf = rf.fit(X_tr,y_tr)
            pred_tr = rf.predict(X_tr)
            pred_val = rf.predict(X_val)
            acc_tr.append(accuracy_score(y_tr,pred_tr))
            acc_val.append(accuracy_score(y_val,pred_val))
            pr_tr.append(precision_score(y_tr,pred_tr))
            pr_val.append(precision_score(y_val,pred_val))
            rec_tr.append(recall_score(y_tr,pred_tr))
            rec_val.append(recall_score(y_val,pred_val))
        fig,ax = plt.subplots(nrows=3,figsize=(10,8))
        plt.sca(ax[0])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],acc_tr,acc_val,'train','val','train/val accuracy')
        plt.sca(ax[1])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],pr_tr,pr_val,'train','val','train/val precision')
        plt.sca(ax[2])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],rec_tr,rec_val,'train','val','train/val recall')
    elif n_estimators:
        for n_estimator in range(n_estimators_start,n_estimators_start+epochs):
            rf = RandomForestClassifier(criterion=criterion,max_depth=6,n_estimators=n_estimator)
            rf = rf.fit(X_tr,y_tr)
            pred_tr = rf.predict(X_tr)
            pred_val = rf.predict(X_val)
            acc_tr.append(accuracy_score(y_tr,pred_tr))
            acc_val.append(accuracy_score(y_val,pred_val))
            pr_tr.append(precision_score(y_tr,pred_tr))
            pr_val.append(precision_score(y_val,pred_val))
            rec_tr.append(recall_score(y_tr,pred_tr))
            rec_val.append(recall_score(y_val,pred_val))
        fig,ax = plt.subplots(nrows=3,figsize=(10,8))
        plt.sca(ax[0])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs)],
               acc_tr,acc_val,'train','val','train/val accuracy')
        plt.sca(ax[1])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs)],
               pr_tr,pr_val,'train','val','train/val precision')
        plt.sca(ax[2])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs)],
               rec_tr,rec_val,'train','val','train/val recall')