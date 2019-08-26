#### Project 2 ---- Exploratory Analysis
#### Analytics Students
#### Hongyang Zheng, Heng Zhou, Zhengqian Xu, Youyou Xie

# Import libraries
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from mlxtend.frequent_patterns import association_rules
from pprint import pprint

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import decomposition
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn import metrics
from sklearn.cluster import DBSCAN 
###############################################################################


#### Main function 
def main(itself):
    
    # load dataset1 and dataset2
    df1=pd.read_csv("data1clean.csv"  )
    df2=pd.read_csv("data2clean.csv")
    
    #### Part 1
   
    # Call function for further cleaning and return new cleaned df1, df2
    df1,df2 = cleaning2(df1, df2) 
    
    # Call function for detecting outliers and return new cleaned df1, df2
    df1,df2 = outlier(df1,df2) 
    
    # Call function for computing statistics after cleaning
    stat(df1,df2)   
    
    # Call function for bining the data
    df1,df2 = bin(df1, df2)
    
    #### Part 2
    # Call function to plot histograms
    histplot(df1,df2)   
    #### First, we need to merge data1 and data2 by symbol
    df3 = pd.merge(df1, df2, left_on='symbol', right_on='Symbol')
    # Call function for correlatoin
    correlation(df3)
    
    #### Part 3
    # Call function for clustering analysis
    clustering(df3)
    
    #### Part 4
    # Call function for association rules
    pre_association_rule(df3)
    
    #### Output df3 for the second part 
    df3.to_csv('dataset3.csv',index=False)
    

###############################################################################
#           Basic Statistical Analysis and Data Cleaning Insight              #
###############################################################################
####1.1 compute stats
# Function to compute statistics
def stat(df1,df2):
    
    # Pick attributes to compute stats
    # Numeric attributes
    df3=pd.concat([df1['avgTotalVolume'],df1['latestPrice'],df1['iexAskPrice'],
                   df1['iexAskSize'],df1['iexBidPrice'],df1['iexBidSize'],
                   df1['ytdChange'],df2['Beta'],df2['MktCap']],axis=1)
    # Categorical attributes
    df4=pd.concat([df1['sector'],df2['industry']],axis=1)
    
    # Create lists to store columns
    columns1=list(df3)
    columns2=list(df4)
    
    # Create lists to store results for attributes
    Mean=[]
    Median=[]
    Sd=[]
    Mode=[]
    
    # Generate statistics for each numeric attribute
    for column in columns1:
        avg=np.nanmean(df3[column])
        Mean.append(avg)
        med = np.nanmedian(df3[column])
        Median.append(med)
        sd=np.nanstd(df3[column])
        Sd.append(sd)
    

    # Determine mode for one categorical variable
    for column in columns2:
        mod=df4.mode()[column][0]
        Mode.append(mod)
        
        
    # Create dictionaries to store results
    dict1={'Mean': Mean, 'Median':Median, 'Standard Deviation':Sd}
    dict2={'Mode':Mode}
    
    # Write results to new dataframes
    df5=pd.DataFrame(dict1,index=columns1)
    df6=pd.DataFrame(dict2, index=columns2)
    
    # Print the results
    print("Results for numerical attributes:")
    pprint(df5)
    print("\n")
    print("Results for categorical attributes:")
    pprint(df6)
    


#### 1.3 Missing values and more cleaning
#### Missing values -- see project2 report
#### Two more cleaning steps
# Define a function for further cleaning
def cleaning2(df1, df2):
      
    #### Clean step one: handle 0 values 
    # Create list stored the name of columns that are needed to be cleaned
    name_list1=('iexAskSize', 'iexBidSize')
    # Use median to fill 0 for these two attributes
    for name in name_list1: 
        replace = df1[df1[name]!=0]
        df1.loc[df1[name] == 0, name] = np.median(replace[name])
    
    # Create list stored the name of columns that are needed to be cleaned
    name_list2=('iexAskPrice', 'iexBidPrice')
    # Use latest price to deal with 0
    for name in name_list2: 
        # Use latestPrice+/- 0.1 to esitimate prices
        df1.loc[df1[name] == 0, name] = df1.loc[df1[name] == 0,'latestPrice']+np.random.uniform(-0.1, 0.1)
    
    # Reindex
    df1=df1.reset_index()
    del df1['index']
    
    
    #### Clean step two: convert categorical variable into numerical variable
    # List categories of Sector
    cate1=list(df1['sector'].unique())
    n1=len(cate1)
    
    # Assign different number to each category
    m=0
    while m < n1:
        for i in range(len(df1.index)):
                if (df1.loc[i,"sector"])==cate1[m] :
                    df1.loc[i,"sector"]=(m+1)
        m=m+1
        
    #### Clean step three: seperate name for industry
    # Seperate name for industry into industry and sub_industry
    df2['industry'], df2['sub_industry'] = df2['industry'].str.split('-', 1).str
    # Fill NA in sub_industry with 'no sub industry'
    df2['sub_industry']=df2['sub_industry'].fillna('no sub')
    
    # drop ';' for sub_industry column
    for i in range(0,len(df2)):
        df2.loc[i,'sub_industry'] = df2.iloc[i,4].replace(';',' ')
        
    # Return new df1, df2
    return df1, df2    
    


#### 1.2 detect outliers
# Function to detect and clean outliers for some attributes    
def outlier(df1,df2):
    
    # Look at the subboxplot for each attribute in two datasets 
    df1.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()
    
    df2.plot(kind='box', subplots=True, layout=(1, 2), sharex=False, sharey=False)
    plt.show()
    
    
    # Extract attributes that are mumeric fot two datasets 
    name_list1 = ('avgTotalVolume','latestPrice','ytdChange','iexAskPrice', 'iexBidPrice')    
    name_list2 = ('Beta','MktCap')
    
    # Detect outlier by IQR and drop them for data
    # For dataset1
    for name in name_list1:
        Q1 = np.percentile(df1[name],25)
        Q3 = np.percentile(df1[name],75)
        IQR = Q3 -Q1
        outlier = 1.5 * IQR
        df1 = df1[np.logical_and(df1[name] > (Q1 - outlier), df1[name]< (Q3 + outlier))]
   
    # For dataset2
    for name in name_list2:
        Q1 = np.percentile(df2[name],25)
        Q3 = np.percentile(df2[name],75)
        IQR = Q3 -Q1
        outlier = 1.5 * IQR
        df2 = df2[np.logical_and(df2[name] > (Q1 - outlier), df2[name]< (Q3 + outlier))]
    
    # For 'iexAskSize', we set a maximum number to drop outliers. 
    df1 = df1[df1['iexAskSize'] < 5000]
    
    # Reindex
    df1=df1.reset_index()
    del df1['index']
    
    # Reindex
    df2=df2.reset_index()
    del df2['index']
    
    # Relook at the subboxplot for each numeric attribute  after dropping outliers     
    df1.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show() 

    df2.plot(kind='box', subplots=True, layout=(1, 2), sharex=False, sharey=False)
    plt.show()
        
    # Return cleaned df1, df2
    return df1, df2
  
       
    
#### 1.4  bin the data
#### Bin numeric variables -- three numerical variables
# Define a function for binning
def bin(df1, df2):
    
    #### 'Beta' in df2
    # Bin 'Beta' into different risk categories according to the value of Beta
    for i in range(len(df2.index)):
        if (df2.loc[i,"Beta"])==1:
            df2.loc[i,"Risk"]='normal'
        if (df2.loc[i,"Beta"])>1:
            df2.loc[i,"Risk"]='big'
        if ((df2.loc[i,"Beta"])<1 and (df2.loc[i,"Beta"])>0):
            df2.loc[i,"Risk"]='small'
        if (df2.loc[i,"Beta"])<0:
            df2.loc[i,"Risk"]='negative'

    #### 'ytdChange' in df1
    # Create bins and label name
    d=pd.Series(df1['ytdChange']).describe()
    bins1=np.sort([d['min']-1,d['25%'],d['mean'],d['50%'],d['75%'],d['max']+1])
    label1=['--','-','|','+','++']
    
    # Bin 'ytdChange' into 5 different categories
    df1['Change_level']=pd.cut(df1['ytdChange'], bins1, labels=label1)
    # Print first 10 rows to see the change
    print("Print first 10 rows to see the change\n")
    print(df1['Change_level'].head(10))
    
    
    #### 'avgTotalVolume' in df1
    # Create bins and label name
    d=pd.Series(df1['avgTotalVolume']).describe()
    bins2=np.sort([d['min']-1,d['25%'],d['mean'],d['50%'],d['75%'],d['max']+1])
    label2=['F','D','C','B','A']
    
    # Bin 'avgTotalVolume' into 5 different categories
    df1['Volume_level']=pd.cut(df1['avgTotalVolume'], bins2, labels=label2)
    # Print first 10 rows to see the change
    print("Print first 10 rows to see the change\n")
    print(df1['Volume_level'].head(10))
    
    #Print first 10 rows of two dataset
    print("\nPrint first 10 rows of two dataset\n")  
    print(df1.head(10))
    print(df2.head(10))
    
    # print data summary using info method
    print("\nDataset1 Summary using info method")
    pprint(df1.info())
    pprint(df1.describe())
    
    print("\nDataset2 Summary using info method")
    pprint(df2.info())
    pprint(df2.describe())
    
    # Return new df1, df2
    return df1, df2  



###############################################################################
#                        Histograms and Correlations                          #
###############################################################################
#### 2.1 
#### Plot for three attributes
# Define a function for histograms
def histplot(df1,df2):
    #### Plot for three attributes
    plt.figure()
    
    # For avgTotalVolume
    plt.subplot(2,2,1) 
    df1['avgTotalVolume'].hist()
    plt.title("Histogram for avgTotalVolume")
    
    # For ytdChange
    plt.subplot(2,2,2) 
    df1['ytdChange'].hist()
    plt.title("Histogram for ytdChange")
   
    # For Beta
    plt.subplot(2,2,3) 
    df2['Beta'].hist()
    plt.title("Histogram for Beta")
    # Show the picture
    plt.show() 
    
    # For MktCap
    plt.subplot(2,2,4) 
    df2['MktCap'].hist()
    plt.title("Histogram for MktCap")
    # Show the picture
    plt.show() 
    
    
#### 2.2
# Define a function for correlatoin   
def correlation(df_new):
       
    # We choose three variables- iexAskPrice,iexBidPrice,ytdChange. And make correlation for these three variables
    df_corr = pd.concat([df_new['MktCap'],df_new['avgTotalVolume'],df_new['latestPrice']],axis = 1)
    df_corr.corr()
    
    plt.figure()
    # Plot the scatterplot
    # For subplot1
    plt.subplot(1,3,1) 
    plt.scatter(df_new['MktCap'],df_new['avgTotalVolume'], s=20, c='b', alpha=.2)
    # Set the title  
    plt.title('Scatter Plot related to MktCap and avgTotalVolume')  
    # Set the x labels
    plt.xlabel('MktCap')  
    # Set the y labels 
    plt.ylabel('avgTotalVolume')  
 
    
    plt.subplot(1,3,2)
    plt.scatter(df_new['MktCap'],df_new['latestPrice'], s=20, c='r', alpha=.5)
    plt.title('Scatter Plot related to MktCap and latestPrice')  
    # Set the x labels
    plt.xlabel('MktCap')  
    # Set the y labels 
    plt.ylabel('latestPrice')    

    
    plt.subplot(1,3,3)
    plt.scatter(df_new['avgTotalVolume'],df_new['latestPrice'], s=20, c='g', alpha=.8)
    plt.title('Scatter Plot related to avgTotalVolume and latestPrice')  
    # Set the x labels
    plt.xlabel('avgTotalVolume')  
    # Set the y labels 
    plt.ylabel('latestPrice') 

    # Show the picture
    plt.show() 


###############################################################################
#                             Cluster Analysis                                #
###############################################################################
#### 3 clustering methods
# Define a function for clusertings, and then define several sub functions  
def clustering(df_new):
    
    # Use min_max normalization to normalize dat
    normalizedDataFrame = normalize(df_new)
    
    # Kmeans Cluestering
    kmeans(normalizedDataFrame)
    
    # Hierarchical Cluestering
    ward(normalizedDataFrame)
    
    # DBSCAN clustering
    dbscan(normalizedDataFrame)


# Use min_max normalization to normalize data        
def normalize(df_new):
    ### First Normalize data 
    # Choose five numeric attributes for   
    df=pd.concat([df_new['sector'],df_new['avgTotalVolume'],df_new['ytdChange'], df_new['Beta'], df_new['MktCap']],axis=1)
    # Returns a numpy array
    x = df.values 
      
    # Transform df into Min-Max Normalization for later calculating
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    return(normalizedDataFrame)

# Kmeans Cluestering    
def kmeans(normalizedDataFrame):   
    
    # Try several numbers for k, to see which one is better
    a = np.arange(2,10,1)
    for k in a:
        kmeans = KMeans(n_clusters=k)
    
        # Cluster and put every row into a cluster
        cluster_labels = kmeans.fit_predict(normalizedDataFrame) 
            
        # Use silhouette_avg and calinski_harabaz_score to determine if the clustering is good
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        calinski_avg = calinski_harabaz_score(normalizedDataFrame, cluster_labels)
        print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
        print("For n_clusters =", k, "The average calinski_harabaz_score is :", calinski_avg)
       
    ### From what above, we can see that when k=3 the clustering reult is the best. 
    ### Reuse kmeans for k=3
    k=3
    kmeans = KMeans(n_clusters=k)
    
    # Cluster and put every row into a cluster
    cluster_labels = kmeans.fit_predict(normalizedDataFrame) # 100 numbers which divided to k groups
            
    # Use calinski_harabaz score to access the quality of data
    calinski_avg = metrics.calinski_harabaz_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters = ", str(k)," the average calinski_harabaz_score is :", calinski_avg)
        
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)

    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("For kmeans method,PCA for k="+str(k))
    plt.savefig("For kmeans method, PCA for k="+str(k))
    plt.show()
    plt.clf()
    plt.close()
    

# Hierarchical Cluestering
def ward(normalizedDataFrame):   
    
    Z = linkage(normalizedDataFrame, method='ward', metric='euclidean')
    
    # Draw the dendrogram to see we should divide to how many cluster
    plt.figure(figsize=(10, 8))
    dendrogram(Z, truncate_mode='lastp', p=20, show_leaf_counts=False, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
    plt.title('Dendrogram for the Agglomerative Clustering')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.show()
    plt.clf()
    
    # From the dendrogram,we can see that 3 clusters does well, so we divide it to 4 clusters
    k = 3
    labels_1 = fcluster(Z, t=k, criterion='maxclust')  

    # Use calinski_harabaz score to access the quality of data
    calinski_avg = metrics.calinski_harabaz_score(normalizedDataFrame, labels_1)
    print("For n_clusters = ", str(k)," the average calinski_harabaz_score is :", calinski_avg)

    #####
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels_1)
    plt.title("For hierarchical method,PCA for k="+str(k))
    plt.savefig("For hierarchical method,PCA for k="+str(k))
    plt.show()
    plt.clf()
    plt.close()   
    
    
# DBSCAN Cluestering    
def dbscan(normalizedDataFrame):
    
    # Set a sery of eps and  min_smpl to see which parameter is the best
    r = np.arange(0.1,0.4,0.04)
    min_smpl = np.arange(2,8,2)
    
    for i in range(0,len(r)):
        for j in range(0,len(min_smpl)):
            clustering = DBSCAN(eps=r[i], min_samples=min_smpl[j]).fit_predict(normalizedDataFrame)
            
            # Use silhouette_avg and calinski_harabaz_score to determine if the clustering is good
            Calin_Hara = metrics.calinski_harabaz_score(normalizedDataFrame, clustering)
            print("ep=",str(r[i]),",min_samples=",str(min_smpl[j]),",The average Calinski-Harabaz is :", Calin_Hara)
            silhouette_avg = silhouette_score(normalizedDataFrame, clustering)
            print("ep=",str(r[i]),",min_samples=",str(min_smpl[j]),", The average silhouette_score is :", silhouette_avg)
    
    
    # eps = 0.14 and min_samples = 6 is the best one, then we use dbscan again
    r = 0.14
    min_smpl = 6    
    clustering = DBSCAN(eps=r, min_samples=min_smpl).fit_predict(normalizedDataFrame)
    # Print(len(np.unique(clustering))) 
    
    # Use calinski_harabaz score to access the quality of data
    Calin_Hara = metrics.calinski_harabaz_score(normalizedDataFrame, clustering)
    print("eps=",str(r),",min_samples=",str(min_smpl),",The average Calinski-Harabaz is :", Calin_Hara)
       
    #####
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    # Using PCA    
    pca2D = decomposition.PCA(2)

    # Turn the data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=clustering)
    plt.title("DBSCAN")
    plt.savefig("DBSCAN")
    plt.show()
    plt.clf()
    plt.close()   
  

###############################################################################
#            Association Rules/Frequent Itemset Mining Analysis               #
###############################################################################
# Define a function for preprocessing data before association rules
def pre_association_rule(df3):
    
    #### Create a subset data
    #### We want to use the variable called 'Risk' in dataset two. 
    # Create subset 
    data=pd.concat([df3['Change_level'], df3['Volume_level'], df3['Risk']], axis=1)
    
    #### Convert data into transactions
    # Used to store transactions list
    transaction=[]
    n=len(data)
    # Convert to transaction list 
    for i in range(0, n):
        transaction.append([str(data.values[i,j]) for j in range(0, 3)])
    # Create transaction
    te = TransactionEncoder()
    record = te.fit(transaction).transform(transaction)
    record=record.astype("int")
    # Create dataframe
    tran=pd.DataFrame(record, columns=te.columns_)
    
    #### Find association rules
    # Four support level
    sup1=0.05
    sup2=0.10
    sup3=0.15
    #sup4=0.20
    
    # Support level one
    association_rule(tran, sup1)
    
    # Support level two
    association_rule(tran, sup2)
    
    # Support level three
    association_rule(tran, sup3)
    
    # Support level four
    # For this level, we don't have any association rule that satisfies the minsup
    # association_rule(tran, sup4)
        

# Define a function for association rules
def association_rule(tran, sup):
    
    #### Get frequent itemsets and calculate support
    print("\n\nWhen minimum supprot is:", sup)
    frequent_itemsets=apriori(tran, min_support=sup, use_colnames=True)
    print("Frequent itemsets:")
    pprint(frequent_itemsets)
    
    # Calculate confidence
    s=association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    s=s.dropna() 
    
    # Adjust the order of columns
    rules = s[['antecedents', 'consequents','support','confidence']]
    print("\nRules:")
    pprint(rules)
    

# Call main function
if __name__ == '__main__':
    main(sys.argv)





