#### Project 2 ---- Predictive Analysis
#### Analytics Students
#### Hongyang Zheng, Heng Zhou, Zhengqian Xu, Youyou Xie

# Import libraries
###############################################################################
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import roc_curve, auc  

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
###############################################################################

#### Main function 
def main(itself):
    
    # load dataset3
    df3=pd.read_csv("dataset3.csv")
    
    # Call function for Parametric Statistical Tests
    testing(df3)
    
    # We use desicion tree and random forest to do the hypothesis1
    hypothesis1(df3)
    
    # Classification method - knn,bayes and svm
    classification(df3)
    
     
###############################################################################
#             Hypothesis Testing -- Parametric Statistical Tests              #
###############################################################################
# Function for Parametric Statistical Tests 
def testing(df3):
    
    # Hypothesis 1 
    # Null hypothesis: the mean of beta for Oil & Gas - Integrated is same as the mean of beta for Oil & Gas - Midstream
    # Alternative hypothesis: the mean of beta for Oil & Gas - Integrated is not same as the mean of beta for Oil & Gas - Midstream
    # Conduct t-test for Beta between Oil&Gas-Integrated and Oil&Gas-Midstream
    OilGas_integrated = df3[df3['sub_industry'] == ' Integrated']['Beta']
    OilGas_midstream = df3[df3['sub_industry'] == ' Midstream']['Beta']
    ttest1=str(stats.ttest_ind(OilGas_integrated, OilGas_midstream))
    pprint(ttest1)
    
    
    # Hypothesis 2
    # Null hypothesis: the mean of ytdChange for 11 sectors are same
    # Alternative hypothesis: the mean of ytdChange for at least one sector of 11 sectors are not same as others
    # Conduct ANOVA test for average total volume between 11 sectors
    mod=ols("ytdChange ~ sector",data=df3).fit()
    table = str(sm.stats.anova_lm(mod, typ=2))
    pprint(table)
    
    
    # Hypothesis 3
    # Null hypothesis: latestPrice and avgTotalVolume are independent
    # Alternative hypothesis: latestPrice and avgTotalVolume are not independent
    # Conduct Linear Regression for latestPrice and avgTotalVolume
    # Linear Regression
    X = df3['latestPrice']
    y = df3['avgTotalVolume']
    model = sm.OLS(y, X).fit()
    pprint(model.summary())
    
    
    # Hypothesis 4
    # Null hypothesis: avgTotalVolume, latestPrice, ytdChange and Beta are independent
    # Alternative hypothesis: at least one variable of latestPrice, ytdChange, Beta is not independent from avgTotalVolume
    # Conduct Linear Regression for latestPrice, ytdChange and avgTotalVolume
    X = df3[['latestPrice','ytdChange','Beta']]
    y = df3['avgTotalVolume']
    model = sm.OLS(y, X).fit()
    pprint(model.summary())


###############################################################################
#               Hypothesis Testing -- Predictive Model Methods                #
###############################################################################
#### The fisrt hypothesis is when classifying, the result of decision tree is the same as the result of random forest.
#### We use desicion tree and random forest to do the hypothesis
def hypothesis1(df_new):
    # Use 'risk' as classification   
    # Select features for analysis
    dfdt= pd.concat([df_new['sector'],df_new['avgTotalVolume'],df_new['latestPrice'], df_new['industry'],df_new['MktCap'],df_new['Risk']],axis=1)
    
    #### Convert categorical variable'industry' into numerical variable
    # List categories of Sector
    cate2=list(dfdt['industry'].unique())
    n2=len(cate2)
    
    # Assign different number to each category
    m=0
    while m < n2:
        for i in range(len(dfdt.index)):
                if (dfdt.loc[i,"industry"])==cate2[m] :
                    dfdt.loc[i,"industry"]=(m+1)
        m=m+1
    
    # Convert class into two category for later ROC curve    
    # dfdt.loc[dfdt['Risk'] == 'small', 'Risk'] = 'positive'
    # dfdt.loc[dfdt['Risk'] == 'big', 'Risk'] = 'positive'
    
    # Returns a numpy array
    x = dfdt.values     
    
    # Choose first 5 columns as attributes and the last column as classification
    X = x[:,0:5]
    Y = x[:,5]
    
    # Choose test size
    test_size = 0.20
    # Set seed
    seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    # Setup 10-fold cross validation to estimate the accuracy of different models
    # Split data into 10 parts
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'
    
    ######################################################
    # Use different algorithms to build models
    ######################################################
    
    # Add each algorithm and its name to the model array
    models = []
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RandomForest', RandomForestClassifier()))

    
    # Evaluate each model, add results to a results array,
    # Print the accuracy results 
    results = []
    names = []
    for name, model in models:
        # k fold
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        
        # Use cross-validation for traning data
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print("\nThe cross_val_score is:\n")
        # Print the result of cross validation
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    #### Here is Validation test 
    #### Decision tree test validation
    print('\n\n********For decision tree, we use test data to validate it :**********')
    cart = DecisionTreeClassifier()
    predictions1 = cart.fit(X_train, Y_train).predict(X_validate)
    
    # Print the accuracy of cart
    print("\nthe accuracy for cart:",accuracy_score(Y_validate, predictions1))
    print("\nthe confusion matrix for cart is:\n",confusion_matrix(Y_validate, predictions1))
    print("\nclassification report for cart:\n",classification_report(Y_validate, predictions1))


    #### Random forest test validation
    print('\n\n********For random forest, we use test data to validate it :**********')
    rf = RandomForestClassifier()
    rf.fit(X_train, Y_train)
    predictions2 = rf.predict(X_validate)
    
    # Print the accuracy of cart
    print("\nthe accuracy for random forest:",accuracy_score(Y_validate, predictions2))
    print("\nthe confusion matrix for random forest is:\n",confusion_matrix(Y_validate, predictions2))
    print("\nclassification report for random forest:\n",classification_report(Y_validate, predictions2))
    
    
    #### Plot ROC Curve
    #### For decision tree
    # Calculate y_score
    y_score1 = cart.fit(X_train, Y_train).predict_proba(X_validate)
     
    #### Compute ROC curve and ROC area for each class
    # For the first label-big
    fpr11,tpr11,_ = roc_curve(Y_validate, y_score1[:,0],pos_label='big')
    roc_auc11 = auc(fpr11,tpr11) 
     
    plt.figure()
    plt.plot(fpr11, tpr11,color='orange',lw=2, label='ROC curve in desicon tree(area = %0.2f)' % roc_auc11) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for decision tree(big label)')
    plt.legend(loc="lower right")
    plt.show()
    
    # For the first label-negative
    fpr12,tpr12,_ = roc_curve(Y_validate, y_score1[:,1],pos_label='negative')
    roc_auc12 = auc(fpr12,tpr12) 
     
    plt.figure()
    plt.plot(fpr12, tpr12,color='orange',lw=2, label='ROC curve in desicon tree(area = %0.2f)' % roc_auc12) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for decision tree(negative label)')
    plt.legend(loc="lower right")
    plt.show()
    
    # For the first label-small
    fpr13,tpr13,_ = roc_curve(Y_validate, y_score1[:,2],pos_label='small')
    roc_auc13 = auc(fpr13,tpr13) 
     
    plt.figure()
    plt.plot(fpr13, tpr13,color='orange',lw=2, label='ROC curve in desicon tree(area = %0.2f)' % roc_auc13) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for decision tree(small label)')
    plt.legend(loc="lower right")
    plt.show()
    
    
    #### For random forest
    y_score2 = rf.fit(X_train, Y_train).predict_proba(X_validate)
     
    #### Compute ROC curve and ROC area for each class
    # For the first label-big
    fpr21,tpr21,_ = roc_curve(Y_validate, y_score2[:,0],pos_label='big') 
    roc_auc21 = auc(fpr21,tpr21) 
     
    plt.figure()
    plt.plot(fpr21, tpr21,color='orange',lw=2, label='ROC curve in random forest(area = %0.2f)' % roc_auc21) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for random forest(big label)')
    plt.legend(loc="lower right")
    plt.show()
    
    # For the first label-negative
    fpr22,tpr22,_ = roc_curve(Y_validate, y_score2[:,1],pos_label='negative') 
    roc_auc22 = auc(fpr22,tpr22) 
     
    plt.figure()
    plt.plot(fpr21, tpr21,color='orange',lw=2, label='ROC curve in random forest(area = %0.2f)' % roc_auc22) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for random forest(negative label)')
    plt.legend(loc="lower right")
    plt.show()
    
    # For the first label-small
    fpr23,tpr23,_ = roc_curve(Y_validate, y_score2[:,2],pos_label='small') 
    roc_auc23 = auc(fpr23,tpr23) 
     
    plt.figure()
    plt.plot(fpr23, tpr23,color='orange',lw=2, label='ROC curve in random forest(area = %0.2f)' % roc_auc23) 
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for random forest(small label)')
    plt.legend(loc="lower right")
    plt.show()


#### Classification method - knn,bayes and svm
def classification(df_new):

    # Convert categorical variable into numerical variable
    Industry1 = list(df_new['industry'].unique())
    n2 = len(Industry1)
    # Assign different number to each category
    m = 0
    while m < n2:
        for i in range(len(df_new.index)):
            if (df_new.loc[i,"industry"])==Industry1[m] :
                df_new.loc[i,"industry"]=(m+1)
        m=m+1
    # Add the new column
    df_new['Volumes'] = None
    
    # Add the values for the new column
    df_new.loc[df_new['Volume_level'] == 'A','Volumes'] = 'High' 
    df_new.loc[df_new['Volume_level'] == 'B','Volumes'] = 'High'
    df_new.loc[df_new['Volume_level'] == 'C','Volumes'] = 'High'    
    df_new.loc[df_new['Volume_level'] == 'D','Volumes'] = 'Low'
    df_new.loc[df_new['Volume_level'] == 'F','Volumes'] = 'Low'
    # Local the numeric values
    x = df_new.values
    X = x[:, [2,4,9,13,14,15]]
    # Local the Volumn_level
    Y = x[:,18]
    # Then make the validation set 20% of the entire
    test_size = 0.20
    # Set seed
    seed = 123
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # Transform df into Min-Max Normalization for later calculating
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    '''
    #normalize the data
    Normalize = preprocessing.scale(X)
    #read the normalized data to the dataframe
    normalizedDataFrame = pd.DataFrame(Normalize)
    '''
    pprint(normalizedDataFrame[:10])


    # Select 10 folds
    num_folds = 10
    seed = 123
    scoring = 'accuracy'
    # Add each algorithm and its name to the model array
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('svm',svm.SVC()))
     
    # Evaluate each model, add results to a results array,
    # Print the accuracy results (remember these are averages and std
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        # Do cross validation
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        # Print the result of these methods
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        

    # Validation test
    ######################################################
    # Make predictions on validation dataset in knn
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions3 = knn.predict(X_validate)
    # Print the accuracy for knn
    print('\n\n********For knn, we use test data to validate it :**********')
    print('The accurary_score of knn is:\n',accuracy_score(Y_validate, predictions3))
    print('The confusion_matrix of knn is:\n',confusion_matrix(Y_validate, predictions3))
    print('The classification of knn is:\n', classification_report(Y_validate, predictions3))
    

    # Make predictions on validation dataset in bayes
    bayes = GaussianNB()
    bayes.fit(X_train, Y_train)
    predictions4 = bayes.predict(X_validate)
    # Print the accuracy for bayes
    print('\n\n********For bayes, we use test data to validate it :**********')
    print('The accurary_score of bayes is:\n',accuracy_score(Y_validate, predictions4))
    print('The confusion_matrix of bayes is:\n',confusion_matrix(Y_validate, predictions4))
    print('The classification of bayes is:\n',classification_report(Y_validate, predictions4))
    
    
    # Make predictions on validation dataset in SVM
    s = svm.SVC()
    s.fit(X_train, Y_train)
    predictions5 = s.predict(X_validate)
    # Print the accuracy for svm
    print('\n\n********For svm, we use test data to validate it :**********')
    print("\nthe accuracy for svm:",accuracy_score(Y_validate, predictions5))
    print("\nthe confusion matrix for svm is:\n",confusion_matrix(Y_validate, predictions5))
    print("\nclassification report for svm:\n",classification_report(Y_validate, predictions5))
    
    
    #### 3 learn to predict each class against each other compute the Y_score
    # knn method
    Y_score3 = knn.fit(X_train,Y_train).predict_proba(X_validate)
    # Compute ROC curve and ROC area for each class
    fpr3,tpr3,threshold3 = roc_curve(Y_validate,Y_score3[:,1],pos_label='Low')
    roc_auc3 = auc(fpr3,tpr3)
    # Plot the figure about the ROC in knn method
    plt.figure()
    plt.plot(fpr3, tpr3, color='orange',lw=2, label='ROC curve in knn(area = %0.2f)' % roc_auc3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for knn')
    plt.legend(loc="lower right")
    plt.show()


    # Bayes method
    Y_score4 = bayes.fit(X_train,Y_train).predict_proba(X_validate)
    # Compute ROC curve and ROC area for each class
    fpr4,tpr4,threshold4 = roc_curve(Y_validate,Y_score4[:,1],pos_label='Low')
    roc_auc4 = auc(fpr4,tpr4)
    # Plot the figure about the ROC in bayes method
    plt.figure()
    plt.plot(fpr4, tpr4, color='orange',lw=2, label='ROC curve in Bayes(area = %0.2f)' % roc_auc4)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for bayes')
    plt.legend(loc="lower right")
    plt.show()


    # svm method
    Y_score5 = s.fit(X=X_train,y=Y_train).decision_function(X_validate)
    # Compute ROC curve and ROC area for each class
    fpr5,tpr5,threshold5 = roc_curve(Y_validate,Y_score5,pos_label='Low')
    roc_auc5 = auc(fpr5,tpr5)
    # Plot the figure about the ROC in svm method
    plt.figure()
    plt.plot(fpr5, tpr5, color='orange',lw=2, label='ROC curve in svm(area = %0.2f)' % roc_auc5)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for bayes')
    plt.legend(loc="lower right")
    plt.show()
    

# Call main function 
if __name__ == "__main__":   
    main(sys.argv)    
    
    
    