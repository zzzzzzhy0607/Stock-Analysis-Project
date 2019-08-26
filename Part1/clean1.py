#### ANLY501 Project Part1

### This is a python program for grading and cleaning dataset1
### Heng Zhou, Hongyang Zheng, Youyou Xie, Zhengqian Xu



# Import necessary libraries
import pandas as pd
from pprint import pprint


# Function for data cleanliness including grade and clean 
def cleanliness():
    
    # Read dataset directly to dataframe
    df1=pd.read_csv("data1.csv")
    # Data summary using info method
    print("Data Summary using info method")
    pprint(df1.info())
    
    # Create a list to store columns name
    print("\n\nColumns of the data")
    columns=list(df1)
    pprint(columns)  
    
    # Call grade function
    grade(df1,columns)
    # Call clean function
    dataset1=Cleaning(df1)    
    
    # Write out cleaned dataset1 to a new csv file
    dataset1.to_csv("data1clean.csv",index=False)
    
# Grade function
def grade(df1,columns):
    
    # Call function to record missing values
    sub1 = missing(df1,columns)    
    # Call function to record zero values in some columns 
    sub2 = noise1(df1,columns)
    # Create a dictionary to store every frac and the grade
    dict3 = {'missingfrac':sub1,'noise1':sub2,'grade':[0]*len(columns)} 
    # Convert the dictionary to a dataframe
    df4 = pd.DataFrame(dict3,index=columns) 
    
    
    ############################################################################
    # This is data quality grading metric 
    # 2 is the weight for missing values, and 4 is the weight for 0 noise values
    df4['grade'] = 100*(1-2*df4['missingfrac']-4*df4['noise1'])
    ############################################################################
    
    # Print the grade table
    print("\n\n The grade of each column" )
    pprint(df4)
    

# Function to count and record missing values 
def missing(df1,columns):
      
    # The number of total rows in the dataframe
    total=len(df1)
    
    # Used to store missing value 
    nanum = []
    frac1=[]
    
    # Calculate missing value fraction for each column
    for column in columns:
        missingvalue=len(df1[df1[column].isnull()])
        nanum.append(missingvalue)
        frac_M=(missingvalue/total)
        frac1.append(frac_M)
    
    # Convert list to a df
    dict1 = {'number':nanum,'fraction':frac1}
    df2 = pd.DataFrame(dict1,index=columns) 
    # Print the table of number and fraction of missing values 
    print("\n\nThe missing value of each column")
    pprint(df2)    
 
    return frac1
 

# Function to count and record 0 noise values
def noise1(df1,columns):
    # Create two lists to store the number of 0 noise values and their fraction
    list1 = [0]*10
    list2 = [0]*10
    
    # Count the total number of rows of the dataset
    total=len(df1)  
    
    # Determine whether there are zero values
    zeronum = sum(df1['avgTotalVolume'] == 0)
    frac_2=(zeronum/total)
    list1[3] = zeronum
    list2[3] = frac_2
    
    # Convert list to a df
    dict2 = {'number':list1,'fraction':list2}
    df3 = pd.DataFrame(dict2,index=columns) 
    # Print the table of number and fraction of 0 noise values 
    print("\n\nThe zero value of each column")
    pprint(df3)
    
    return list2


# Function to clean the problematic columns 
def Cleaning(df1):
    
    # Drop 0 noise value for this column
    df2 = df1[df1['avgTotalVolume'] != 0]
    # Drop all missing values
    df2 = df2.dropna()
    
    # Print data summary information after cleaning 
    print("\n\n**********************************")
    print("\nAfter cleaning, regrade our data")
    print("\nData Summary using info method")
    pprint(df2.info())
    
    # Regrade after cleaning
    columns=list(df2)
    grade(df2,columns)


    return df2

cleanliness()

