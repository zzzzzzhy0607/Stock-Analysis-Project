#### ANLY501 Project Part3
#### Analytics Students
#### Heng Zhou, Hongyang Zheng, Youyou Xie, Zhengqian Xu



#### Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community
import plotly
import sys
plotly.tools.set_credentials_file(username='visual2018', api_key='GjTRZnRy4kObP5vxUKcZ')
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
#import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import decomposition
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn import metrics
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


#### Main function
def main(itself):
    
    #### Read data
    df=pd.read_csv("dataset3.csv")
    
    #### Data preparation: average value of each numeric attributes for 11 sectors
    gp=df.groupby(by=['sector']).mean()
    
    #### Extra exploratory analysis to determine attributes we are going to use
    exploratory(df)
    
    #### Function for creating the network analysis object
    create_network(df)
    
    #### Extra clustering analysis
    clustering(df)
    
    #### Functions for visualization
    # 3-D scatter plot by colors and markers
    visualization1(df)
    # Boxplot 
    visualization2(df)   
    # Scatter plot
    visualization3(df)
    # Scatter plot
    visualization4(df)
    # Heatmap
    visualization5(df)
    # Heatmap
    visualization6(df)
    # Heatmap
    visualization7(df)
    # Bar plot
    visualization8(df,gp)  
    
    # Visualizations 9-11 with Tableau Public
    # https://public.tableau.com/profile/heng.zhou5950#!/vizhome/project_part3/Project_Part3
 
    
###############################################################################
#                        Extra  Exploratory Analysis                          #
###############################################################################
#### Function for extra exploratory analysis
def exploratory(df):
   
    #### Correlation between 5 numerical attributes 
    df_corr = pd.concat([df['MktCap'],df['avgTotalVolume'],df['latestPrice'],df['Beta'],df['ytdChange']],
                        axis = 1)
    a=df_corr.corr()
    print(a)
    
    
    #### Plot histograms for average total volume, latest price and market capitalization
    # Histogram of average total volume
    plt.hist(df['avgTotalVolume'], color = 'yellow', edgecolor = 'black')
    plt.title('Histogram of average total volume')
    plt.xlabel('Average Total Volume')
    plt.ylabel('Frequency')
    plt.show()
    
    # Histogram of latest price
    plt.hist(df['latestPrice'], color = 'green', edgecolor = 'black')
    plt.title('Histogram of latest price')
    plt.xlabel('Latest Price')
    plt.ylabel('Frequency')
    plt.show()
    
    # Histogram of market capitalization
    plt.hist(df['MktCap'], color = 'gray', edgecolor = 'black')
    plt.title('Histogram of market capitalization')
    plt.xlabel('Market Capitalization')
    plt.ylabel('Frequency')
    plt.show()
    
    
    #### Conduct ANOVA test for average total volume and latest price between 11 sectors    
    # Null hypothesis: the average total volume for 11 sectors are same
    # Alternative hypothesis: the average total volume for at least one sector of 11 sectors are significantly different
    print('\nANOVA test for average total volume:')
    mod_a=ols("avgTotalVolume ~ sector",data=df).fit()
    table_a = str(sm.stats.anova_lm(mod_a, typ=2))
    print(table_a)
    
    # Null hypothesis: the latest price for 11 sectors are same
    # Alternative hypothesis: the latest price for at least one sector of 11 sectors are significantly different
    print('\nANOVA test for latest price:')    
    mod_p=ols("latestPrice ~ sector",data=df).fit()
    table_p = str(sm.stats.anova_lm(mod_p, typ=2))
    print(table_p)
    



###############################################################################
#                              Network Analysis                               #
###############################################################################
#### Function for creating the network analysis object
def create_network(df):
    
    ########################## 
    ## Create Edge and Node ##
    ##########################
    
    #### In our project, edge is 'sector' and node is 'industry' 
    column_edge = 'sector'
    column_ID = 'industry'
    
    # Select columns, remove NaN
    data_to_merge = df[[column_ID, column_edge]].dropna(subset=[column_edge]).drop_duplicates() 
    
    #### To create connections between 'industry' with same 'sector'
    # Join data with itself on the 'industry' column.
    data_to_merge = data_to_merge.merge(
        data_to_merge[[column_ID, column_edge]].rename(columns={column_ID:column_ID+"_2"}), 
        on=column_edge)
        
    # By joining the data with itself, 'industry' will have a connection with themselves.
    # Remove self connections, to keep only connected 'industry' that are different.
    d = data_to_merge[~(data_to_merge[column_ID]==data_to_merge[column_ID+"_2"])] \
        .dropna()[[column_ID, column_ID+"_2", column_edge]]
        
    # To avoid counting twice the connections ('industry1' connected to 'industry2'  and 'industry2' connected to 'industry1' )
    # We force the first 'industry' cloumn to be "lower" than 'industry' cloumn_2
    d.drop(d.loc[d[column_ID+"_2"]<d[column_ID]].index.tolist(), inplace=True)
    
 
    ######################### 
    ##   Create Network    ##
    ######################### 
    
    myNetXGraph = nx.from_pandas_edgelist(df=d, source=column_ID, target=column_ID+'_2', edge_attr=column_edge)   
    myNetXGraph.add_nodes_from(nodes_for_adding=df['industry'].tolist())
    
    nx.draw(myNetXGraph,node_color='green')
    plt.show()
    plt.savefig('nxdraw')
    plt.clf()
    
    
    #### Function for basic information about network
    basicstat(myNetXGraph)



#### Function for basic information about network
def basicstat(myNetXGraph):
    
    ###########################################
    ##    Basic Information about Network    ##
    ###########################################
    
    #### Compute and print the edges and nodes of the this network   
    print("Some basic stat for this network")
    nbr_nodes = nx.number_of_nodes(myNetXGraph)
    nbr_edges = nx.number_of_edges(myNetXGraph)
    print("\nNumber of nodes:", nbr_nodes)
    print("\nNumber of edges:", nbr_edges)
    

    ###########################################
    ##  Local Metrics and Centrality Metrics ##
    ###########################################
    
    print("\n*********************************")
    print("\nLocal Metrics and Centrality Metrics:")
    
    
    ################################# betweenness #################################
    ### Compute the betweenness and make a plot for network based on betweenness
    betweenList = nx.betweenness_centrality(myNetXGraph)
    print("\nBetweeness of each node")
    print(betweenList)
    
    ### sort the dictionary according to value
    print("\nSort the betweenness for each node:")
    print(sorted(betweenList.items(),key = lambda d:d[1],reverse=True))
    
    # Create a plot that colors nodes based on betweenness
    value1 = [betweenList.get(node) for node in myNetXGraph.nodes()]
    avebetween = sum(value1)/len(value1)
    print("\nAverage betweeness:", avebetween,'\n')
    
    # draw the graph
    nx.draw_spring(myNetXGraph, cmap = plt.get_cmap('jet'), node_color = value1, node_size=100, with_labels=False)
    plt.show()
    plt.savefig('Betweeness')
    plt.clf()
      
    
    ################################# degree #################################
    ### Compute the degree and make a plot for network based on degree
    degreeList = nx.degree_centrality(myNetXGraph)
    print("\n Degree of each node:")
    print(degreeList)  
    
    # Create a plot that colors nodes based on degree
    value2 = [degreeList.get(node) for node in myNetXGraph.nodes()]
    avedegree = sum(value2)/len(value2)
    print("\nAverage degree:", avedegree,'\n')
    
    ### sort the dictionary according to value
    print("\nSort the degree for each node:")
    print(sorted(degreeList.items(),key = lambda d:d[1],reverse=True))
    
    # draw the graph
    nx.draw_spring(myNetXGraph, cmap = plt.get_cmap('jet'), node_color = value2, node_size=100, with_labels=False)
    plt.show()
    plt.savefig('degree')
    plt.clf()
    
       
    ################################# eigenvector #################################
    ### Compute the eigenvector and make a plot for network based on eigenvector
    eigenList = nx.eigenvector_centrality(myNetXGraph)
    print("\n Eigenvector of each node:")
    print(eigenList)  
    
    # Create a plot that colors nodes based on eigenvector
    value3 = [eigenList.get(node) for node in myNetXGraph.nodes()]
    aveeigen = sum(value3)/len(value3)
    print("\nAverage eigenvector:",aveeigen,'\n')
    
    ### sort the dictionary according to value
    print("\nSort the eigenvector for each node:")
    print(sorted(eigenList.items(),key = lambda d:d[1],reverse=True))
    
    nx.draw_spring(myNetXGraph, cmap = plt.get_cmap('jet'), node_color = value3, node_size=100, with_labels=False)
    plt.show()
    plt.savefig('eigenList')
    plt.clf()
       
    
    
    ################################# closeness #################################
    ### Compute the closeness and make a plot for network based on closeness
    closeList = nx.closeness_centrality(myNetXGraph)
    print("\n Closeness of each node:")
    print(closeList)  
    
    # Create a plot that colors nodes based on closeness
    value4 = [closeList.get(node) for node in myNetXGraph.nodes()]
    aveclose = sum(value4)/len(value4)
    print("\nAverage closeness:",aveclose,'\n')
    
    ### sort the dictionary according to value
    print("\nSort the closeness for each node:")
    print(sorted(closeList.items(),key = lambda d:d[1],reverse=True))
    
    nx.draw_spring(myNetXGraph, cmap = plt.get_cmap('jet'), node_color = value4, node_size=100, with_labels=False)
    plt.show()
    plt.savefig('closeness')
    plt.clf()
    
    ### Compute the clustering coefficient
    print("\n clustering coefficient of each node:")
    print(nx.clustering(myNetXGraph))
    
    
    ########################
    ##   Global Metrics   ##
    ########################
    
    print("\n*********************************")
    print("\nGlobal Metrics:")
    
    # Compute the density of the network
    density = nbr_edges/(nbr_nodes*(nbr_nodes-1)/2)
    print("\nDensity of this network:", density)
    
    # Computer the number of triangles
    nbr_tranList = nx.triangles(myNetXGraph)
    print("\nEach node's triangle:",nbr_tranList)
    
    # Calculate the total number of triangles
    value = [nbr_tranList.get(node) for node in myNetXGraph.nodes()]
    
    # Since the number of triangles should be divided by 3 due to the duplicates
    nbr_tran = sum(value)/3
    print("\nThe number of triangles:",nbr_tran)


    ################################
    ##   Clustering Analysis      ##
    ################################
    
    print("\n*********************************")
    print("\nClustering analysis:") 
    
    # Conduct modularity clustering
    partition = community.best_partition(myNetXGraph)

    # Print clusters (You will get a list of each node with the cluster you are in)
    print("\nClusters")
    print(partition)

    # Get the values for the clusters and select the node color based on the cluster value
    values2 = [partition.get(node) for node in myNetXGraph.nodes()]
    
    # Determine how many clusters
    nbr_cluster = len(np.unique(values2))
    print("\nThe number of clusters:",nbr_cluster,'\n')
    
    # View the plot
    nx.draw_spring(myNetXGraph, cmap = plt.get_cmap('jet'), node_color = values2, node_size=100, with_labels=False)
    plt.show()
    plt.savefig('clustering')
    plt.clf()
     
    # Determine the final modularity value of the network
    modValue = community.modularity(partition,myNetXGraph)
    print("\nmodularity:", modValue)
    print("\n***********************************")  



###############################################################################
#                        Extra  Clustering Analysis                           #
###############################################################################
# Define a function for clusertings, and then define several sub functions  
def clustering(df_new):
    
    # Use min_max normalization to normalize dat
    normalizedDataFrame = normalize(df_new)
    
    # Kmeans Cluestering
    kmeans(normalizedDataFrame,df_new)
    
    # Hierarchical Cluestering
    ward(normalizedDataFrame,df_new)
   

# Use min_max normalization to normalize data        
def normalize(df_new):
    ### First Normalize data 
    # Choose five numeric attributes for   
    df=pd.concat([df_new['sector'],df_new['ytdChange'], df_new['Beta'],df_new['avgTotalVolume']],axis=1)
    # Returns a numpy array
    x = df.values 
      
    # Transform df into Min-Max Normalization for later calculating
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    
    return(normalizedDataFrame)


# Kmeans Cluestering    
def kmeans(normalizedDataFrame,df_new):   
    
    ### set kmeans for k=5
    k=5
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

    
    ## Get labels for each sector
    print("\n**************Kmeans**************\n")
    df_new['labels'] = cluster_labels
    dfa = pd.concat([df_new['sector'],df_new['labels']],axis = 1)
    for i in range(0,5):
        dfa0 = dfa.loc[dfa['labels'] == i,:]

        
        print("For cluster",i)
        print(dfa0['sector'].value_counts())


# Hierarchical Cluestering
def ward(normalizedDataFrame,df_new):

    Z = linkage(normalizedDataFrame, method='ward', metric='euclidean')
    
    # we set k=5
    k = 5
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

    
    ## Get labels for each sector
    print("\n**************Hierarchical**************\n")
    df_new['labels'] = labels_1
    dfa = pd.concat([df_new['sector'],df_new['labels']],axis = 1)
    for i in range(1,6):
        dfa0 = dfa.loc[dfa['labels'] == i,:]

        
        print("For cluster",i)
        print(dfa0['sector'].value_counts())



###############################################################################
#                                  Visualization                              #
###############################################################################
# Visualizaiton 1
# 3-D scatter plot by colors and markers
def visualization1(df):
    
    # Create a 3D fig
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color and mark differently for each sector
    for i,c,m in [(1,'k','o'),(2,'r','v'),(3,'c','p'),(4,'g','*'),(5,'b','+'),(6,'y','X'),(7,'m','D'),(8,'pink','H'),(9,'navy','^'),(10,'orange','x'),(11,'violet','1')]:
        x = df.loc[df['sector'] == i,'latestPrice']
        y = df.loc[df['sector'] == i,'ytdChange']
        z = df.loc[df['sector'] == i,'Beta']
        ax.scatter(x, y, z, c=c, marker=m)
    
    # Set labels for each dimension    
    ax.set_xlabel('Latest Price')
    ax.set_ylabel('Year Change of Stock Price')
    ax.set_zlabel('Risk Coefficient Beta')
    
    # Show the plot
    plt.show()



# Visualization 2
# Boxplot of year change for different kinds of risk level-'small','big','negative'
def visualization2(df):

    # Create a trace for boxplot
    # for risk 'small'
    trace0 = go.Box(
        y=df.loc[df['Risk'] == 'small','ytdChange'],
        name = 'year change for small risk stocks',
        marker = dict(
            color = 'rgb(12, 12, 140)',
        )
    )
        
    # Create a trace for boxplot
    # for risk 'big'
    trace1 = go.Box(
        y=df.loc[df['Risk'] == 'big','ytdChange'],
        name = 'year change for big risk stocks',
        marker = dict(
            color = 'red',
        )
    )
    
    # Create a trace for boxplot
    # for risk 'negative'
    trace2 = go.Box(
        y=df.loc[df['Risk'] == 'negative','ytdChange'],
        name = 'year change for negative risk stocks',
        marker = dict(
            color = 'rgb(12, 100, 128)',
        )
    ) 
    
    
    # Assign it to an iterable object named myData    
    myData = [trace0,trace1,trace2]
    
    # Add axes and title
    myLayout = go.Layout(
    	title = "Boxplot of year change for different kinds of risk level",
    	xaxis=dict(
    		title = 'risk level'
    	),
    	yaxis=dict(
    		title = 'year change'
    	)
    )
    
    # Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)
    
    # Create the boxplot
    py.plot(myFigure, filename='Boxplot of year change for different kinds of risk level')



# Visualization 3
# Scatterplot of average total volume and market capitalization for all stocks
def visualization3(df):
   
    # Create a trace for scatter plot
    trace0 = go.Scatter(
            x = df['avgTotalVolume'],
            y = df['MktCap'],
            mode = 'markers',
            )
            
    # Assign it to an iterable object named myData
    myData = [trace0]
    
    # Add axes and title
    myLayout = go.Layout(
     title = "Average Total Volume and Market Capitalization",
     xaxis=dict(
    	 title = 'Average Total Volume'
     ),
     yaxis=dict(
    	 title = 'Market Capitalization'
     )
    )
    
    # Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)
    
    # Create the scatterplot
    py.plot(myFigure, filename='Scatter plot of Average Total Volume and Market Capitalization')
    

    
# Visualization 4
# Scatterplot of beta and market capitalization for 11 sectors      
def visualization4(df):
    
    # Create a trace for scatter plot
    trace0 = go.Scatter(
            x = df['Beta'],
            y = df['MktCap'],
            mode = 'markers',
            marker=dict(
                    size=10,
                    color = df['sector'], #set color equal to a variable
                    colorscale='Viridis',
                    colorbar = dict(
                        title = 'Sector',
                        titleside = 'top',
                        tickmode = 'array',
                        tickvals = [1,2,3,4,5,6,7,8,9,10,11],
                        ticktext = ['Energy','Consumer Cyclical','Communication Services','Real Estate'
                                    ,'Technology','Industrials','Healthcare','Financial Services',
                                    'Basic Materials','Consumer Defensive','Utilities'],
                        ticks = 'inside'),                    
                    showscale=True))

    # Assign it to an iterable object named myData   
    myData = [trace0]
    
    # Add axes and title
    myLayout = go.Layout(
     title = "Beta and Market Capitalization among 11 sectors",
     xaxis=dict(
    	 title = 'Beta'
     ),
     yaxis=dict(
    	 title = 'Market Capitalization'
     )
    )
    
    # Setup figure
    myFigure = go.Figure(data=myData, layout=myLayout)
    
    # Create the scatterplot
    py.plot(myFigure, filename='Scatter plot of Beta and Market Capitalization among 11 sectors')
    
    
    
# Visualization 5
# Heatmap of latest price for 11 sectors    
def visualization5(df):
       
    # Create a trace for heatmap
    trace0 = go.Heatmap(z=[df.loc[df['sector'] == 1,'latestPrice'], 
                           df.loc[df['sector'] == 2,'latestPrice'],
                           df.loc[df['sector'] == 3,'latestPrice'],
                           df.loc[df['sector'] == 4,'latestPrice'],
                           df.loc[df['sector'] == 5,'latestPrice'],
                           df.loc[df['sector'] == 6,'latestPrice'],
                           df.loc[df['sector'] == 7,'latestPrice'],
                           df.loc[df['sector'] == 8,'latestPrice'],
                           df.loc[df['sector'] == 9,'latestPrice'],
                           df.loc[df['sector'] == 10,'latestPrice'],
                           df.loc[df['sector'] == 11,'latestPrice']],
                        y = ['Energy','Consumer Cyclical','Communication Services','Real Estate'
                             ,'Technology','Industrials','Healthcare','Financial Services',
                             'Basic Materials','Consumer Defensive','Utilities'])
    
    # Assign it to an iterable object named myData   
    myData=[trace0]
    
    # Create the heatmap
    py.plot(myData, filename='Heatmap of latest price for each sector')
 
 
    
# Visualization 6
# Heatmap of average total volume of 3 different risk level-'small','big','negative' for 11 sectors     
def visualization6(df):
    
    # Create 3 lists of average total volume for 3 risk levels
    #For big risk
    Risk1=[]
    for i in range(11):
        df1=df.loc[(df['sector'] == i+1) & (df['Risk']=='big')]
        Risk1.append(np.mean(df1['avgTotalVolume']))
        
    #For small risk    
    Risk2=[]
    for i in range(11):
        df2=df.loc[(df['sector'] == i+1) & (df['Risk']=='small')]
        Risk2.append(np.mean(df2['avgTotalVolume'])) 
        
    #For negative risk
    Risk3=[]
    for i in range(11):
        df3=df.loc[(df['sector'] == i+1) & (df['Risk']=='negative')]
        Risk3.append(np.mean(df3['avgTotalVolume']))
        
        
    # Create a trace for heatmap  
    trace0 = go.Heatmap(z=[Risk1,Risk2,Risk3],
                        x = ['Energy','Consumer Cyclical','Communication Services',
                             'Real Estate','Technology','Industrials','Healthcare',
                             'Financial Services','Basic Materials','Consumer Defensive','Utilities'],
                        y = ['big','small','negative'])
    
    # Assign it to an iterable object named myData   
    myData=[trace0]
    
    # Create the heatmap
    py.plot(myData, filename='Heatmap of Average Total Volume under different risk levels for 11 Sectors')    


  
# Visualization 7
# Heatmap of average total volume of 5 different change level-'++','+','|','-','--' for 11 sectors     
def visualization7(df):
    
    # Create 5 lists of average total volume for 5 change levels
    # For ++ level
    Change1=[]
    for i in range(11):
        df1=df.loc[(df['sector'] == i+1) & (df['Change_level']=='++')]
        Change1.append(np.mean(df1['avgTotalVolume']))
        
    # For + level   
    Change2=[]
    for i in range(11):
        df1=df.loc[(df['sector'] == i+1) & (df['Change_level']=='+')]
        Change2.append(np.mean(df1['avgTotalVolume']))
        
    # For | level
    Change3=[]
    for i in range(11):
        df1=df.loc[(df['sector'] == i+1) & (df['Change_level']=='|')]
        Change3.append(np.mean(df1['avgTotalVolume']))
    
    # For - level    
    Change4=[]
    for i in range(11):
        df1=df.loc[(df['sector'] == i+1) & (df['Change_level']=='-')]
        Change4.append(np.mean(df1['avgTotalVolume']))
    
    # For -- level    
    Change5=[]
    for i in range(11):
        df1=df.loc[(df['sector'] == i+1) & (df['Change_level']=='--')]
        Change5.append(np.mean(df1['avgTotalVolume']))
       
        
    # Create a trace for heatmap    
    trace0 = go.Heatmap(z=[Change1,Change2,Change3,Change4,Change5],
                        x = ['Energy','Consumer Cyclical','Communication Services',
                             'Real Estate','Technology','Industrials','Healthcare',
                             'Financial Services','Basic Materials','Consumer Defensive','Utilities'],
                        y = ['++','+','|','-','--'])
    
    # Assign it to an iterable object named myData   
    myData=[trace0]
    
    # Create the heatmap
    py.plot(myData, filename='Heatmap of Average Total Volume under different change levels for 11 Sectors')    
    

    
# Visualization 8
# Bar plot of average latest price, IEX website ask price and IEX website bid price for 11 sectors     
def visualization8(df,gp):
    
    # Create a trace for bar plot
    trace0 = go.Bar(
        x=['Energy','Consumer Cyclical','Communication Services','Real Estate',
           'Technology','Industrials','Healthcare','Financial Services',
           'Basic Materials','Consumer Defensive','Utilities'],
        y=gp['latestPrice'],
        name='Latest Price',
        marker=dict(
            color='rgb(55, 83, 109)'))
        
    # Create a trace for bar plot  
    trace1 = go.Bar(
        x=['Energy','Consumer Cyclical','Communication Services','Real Estate',
           'Technology','Industrials','Healthcare','Financial Services',
           'Basic Materials','Consumer Defensive','Utilities'],
        y=gp['iexAskPrice'],
        name='IEX website Ask Price',
        marker=dict(
            color='rgb(26, 118, 255)'))
        
    # Create a trace for bar plot        
    trace2 = go.Bar(
        x=['Energy','Consumer Cyclical','Communication Services','Real Estate',
           'Technology','Industrials','Healthcare','Financial Services',
           'Basic Materials','Consumer Defensive','Utilities'],
        y=gp['iexBidPrice'],
        name='IEX website Bid Price',
        marker=dict(
            color='rgb(26, 120, 155)'))
    
    # Assign it to an iterable object named myData   
    myData = [trace0, trace1,trace2]
    
    # Add axes and title
    layout = go.Layout(
        title='Three Average Prices for Each Sector',
        xaxis=dict(
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title='USD',
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    # Setup figure
    fig = go.Figure(data=myData, layout=layout)
    # Create the bar plot
    py.plot(fig, filename='Bar plot of three average prices for each sector')
    
           
#### Call for main
if __name__ == "__main__":
    main(sys.argv)   
    
    
    