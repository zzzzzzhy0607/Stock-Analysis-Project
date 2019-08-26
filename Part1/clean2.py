#### ANLY501 Project Part1

### This is a python program for grading and cleaning dataset2
### Heng Zhou, Hongyang Zheng, Youyou Xie, Zhengqian Xu



# Import necessary libraries
import pandas as pd
from pprint import pprint


# Function for data cleanliness including grade and clean 
def cleanliness():
    
    # Read dataset directly to dataframe
    df1=pd.read_csv("data2.csv")
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
    dataset2 = Cleaning(df1)
    
    # Write out cleaned dataset2 to a new csv file 
    dataset2.to_csv("data2clean.csv",index=False)
   
# Grade function
def grade(df1,columns):
    
    # Call function to record missing values
    sub1 = missing(df1,columns)    
    # Call function to record zero values in some columns 
    sub2 = noise1(df1,columns)
    sub3 = noise2(df1,columns)
    
    # Create a dictionary to store every frac and the grade
    dict3 = {'missingfrac':sub1,'noise1':sub2,'noise2':sub3,'grade':[0]*len(columns)}   
    df4 = pd.DataFrame(dict3,index=columns)  
    
    
    
    ############################################################################
    # This is data quality grading metric 
    # 2 is the weight for missing values, and 4 is the weight for 0 noise values
    df4['grade'] = 100*(1-2*df4['missingfrac']-4*df4['noise1']-4*df4['noise2'])
    ############################################################################
    
    # Print the grade table
    print("\n\n The grade of each value" )
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
    
    # The number of total rows in the dataframe
    total=len(df1)
    
    # Create two lists to store zero value     
    zeronum = []
    frac2 = []
    
    # Determine if there are zero values
    for column in columns:
        zerovalue = sum(df1[column] == 0)
        zeronum.append(zerovalue)
        frac_Z=(zerovalue/total)
        frac2.append(frac_Z)
    
    # Convert list to a df
    dict2 = {'number':zeronum,'fraction':frac2}
    df3 = pd.DataFrame(dict2,index=columns) 
    # Print the table of number and fraction of 0 noise values 
    print("\n\nThe noise1(zero) value of each column")
    pprint(df3)
    
    
    return frac2

# Function to count noise value for df['Beta'] 
def noise2(df1,columns):
    
    # The number of total rows in the dataframe
    total=len(df1)
    
    # Determine if there are value of Beta > 100
    Over100= sum(df1['Beta'] >= 100)
    frac_3 = (Over100/total)
    
    # Convert list to a df
    dict3 = {'number':[0,Over100,0,0],'fraction':[0,frac_3,0,0]}
    df4 = pd.DataFrame(dict3, index = columns)
    # Print the table of number and fraction of 0 noise values 
    print("\n\nThe noise2(number over 100) value of each column")
    pprint(df4)

    return [0,frac_3,0,0]


# Function to clean the problematic columns 
def Cleaning(df1):
    
    # Drop invalid value
    df2 = df1[df1['Beta']!=0]
    df2 = df2[df2['MktCap']!=0]
    df2 = df2[df2['Beta'] < 100]
    
    # Replace NaNs in industry
    df2.loc[df2['Symbol']=='HFRO','industry'] = 'Bank'
    df2.loc[df2['Symbol']=='IHTA','industry'] = 'Asset Management'
    df2.loc[df2['Symbol']=='DCF','industry'] = 'Advertising & Marketing Services'
    
    # Drop missing value
    df2 = df2.dropna()
    
    # Regrade after cleaning
    print("\n\n*****************************************")
    print("\n\nAfter cleaning, regrade our data")
    print("\n\nData Summary using info method")
    pprint(df2.info())
    
    columns=list(df2)
    grade(df2,columns)
    
    return df2

cleanliness()

