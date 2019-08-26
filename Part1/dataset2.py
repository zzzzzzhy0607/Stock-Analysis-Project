#### ANLY501 Project Part1

### This is a python program for scraping dataset2
### Heng Zhou, Hongyang Zheng, Youyou Xie, Zhengqian Xu

# Import necessary libraries
import requests
import json
from bs4 import BeautifulSoup
import pandas as pd


# Function to scrape dataset two which contains company and market information 
def dataset2():
    
    # Create a csv file to store the dataset
    f1 = open("dataset2.csv","w")
    f1.close()
    
    # Used to storing sector name 
    data=[] 
    # Read sector name from the input file
    with open("input_symbol.txt","r") as f0:  
        symbols=f0.readlines()
    
    for line in symbols:
        line=line.replace('\n','')
        data.append(line)
    
    # Call scrape function to gather data
    scrape(f1, data)
        

# Function that uses URL to scrape data
def scrape(f1, data):    
    
    # URL post            
    for i in range(0,len(data)):
        print(data[i])
        # BaseURL
        BaseURL= "https://financialmodelingprep.com/api/company/profile/%s" % data[i]
        
        # Load data as json format
        response = requests.get(BaseURL)
        content = response.content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Determine whether the symbol is a valid input 
        if(soup.find_all('pre')!= []):
            preTag = soup.find_all('pre')[0].text
            jsonData = json.loads(preTag)
            # Create dataframe for each output 
            df=pd.DataFrame(jsonData).T
            
            # Write out the dataframe into a csv file
            with open("dataset2.csv","a",encoding="utf8") as f1:
                df.to_csv(f1,index=False) 
                
        # Reread the dataframe for variable selection
        df = pd.read_csv("dataset2.csv",header=0)
    # Call function to drop duplicate rows and generate final dataset2
    generate(df)


# Function to drop duplicate rows and generate final dataset2
def generate(df):
    
    # Drop the duplicate rows
    df = df.drop_duplicates(keep=False) 
    # Final dataset2
    df1 = pd.concat([df['Unnamed: 0.1'],df['Beta'],
                     df['MktCap'],df['industry']],axis=1)
    # Rename the first column
    df1.rename(columns={'Unnamed: 0.1': "Symbol" }, inplace=True)
    df1.to_csv("data2.csv",index=False)       


dataset2()









