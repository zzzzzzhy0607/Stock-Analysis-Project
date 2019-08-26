#### ANLY501 Project Part1

### This is a python program for scraping dataset1
### Heng Zhou, Hongyang Zheng, Youyou Xie, Zhengqian Xu

# Import necessary libraries
import requests
import pandas as pd

# Function for data scraping from a stock exchange website
def dataset1():
    
    # Create a list to storing sector name 
    data=[] 
    # Read sector name from the input file
    with open("input_sector.txt","r") as f0:  
        sector=f0.readlines()
    
    for line in sector:
        line=line.replace('\n','')
        data.append(line)
     
    # Create an output file to write the results
    f1 = open("dataset1.csv","w")
    f1.close()
    
    # Call scrape function to gather data
    scrape(f1, data)
   
    
# Function that uses URL to scrape data
def scrape(f1, data): 
    # An example of the URL we used to extract the data 
    # https://api.iextrading.com/1.0/stock/market/collection/sector?collectionName=Health%20Care
    # BaseURL
    BaseURL="https://api.iextrading.com/1.0/stock/market/collection/sector"
    
    # URL post            
    for i in range(0,len(data)):
        URLPost = {'collectionName': data[i]}
        # load data as json format
        response=requests.get(BaseURL, URLPost)
        jsontxt = response.json()
        
        # Convert to a dataframe
        df=pd.DataFrame(jsontxt)
        
        # Write out the data to a csv file 
        with open("dataset1.csv","a",encoding="utf8") as f1:
            df.to_csv(f1, index=False)
            f1.close()
    
    # Read dataset again
    df = pd.read_csv("dataset1.csv",header=0)
    
    # Call function to drop duplicate rows and generate final dataset2
    generate(df)
    

# Function to drop duplicate rows and generate final dataset2
def generate(df):
      
    # Drop the duplicate rows
    df=df.drop_duplicates(keep=False) 
    
    # Final dataset1
    df1 = pd.concat([df['companyName'],df['symbol'],df['sector'],df['avgTotalVolume'],df['latestPrice'],
                     df['iexAskPrice'],df['iexAskSize'],df['iexBidPrice'],df['iexBidSize'],
                     df['ytdChange']],axis=1)
    
    # Write out final dataset to csv file
    df1.to_csv("data1.csv",index=False) 

    # Extract symbol from data1 as input_symbol for dataset2 
    df2=df['symbol'] 
    df2.to_csv("input_symbol.txt",index=False) 


dataset1()



