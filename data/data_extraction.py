import sqlite3
import re
import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup




if __name__ == "__main__":
    con = sqlite3.connect("data/database.sqlite")
    data = pd.read_sql_query('SELECT * FROM reviews',con)

    time_out_secs = 60
    # for chrome, uncomment the next line and comment this one.
    driver = webdriver.Edge("web drivers/msedgedriver.exe")
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.implicitly_wait(time_out_secs)
    driver.set_page_load_timeout(time_out_secs)
    
    file_path = "D:/CUNY/NLP/project/Album-Reviews-Sentiment-Analysis/data/final_data.csv"
    for i,row in data.iloc[:].iterrows():
        try:    
            driver.get(row[3])
            time.sleep(1.5) # Waiting for the driver to completely open the website
            page_source = driver.page_source
            # The review text in the website is written in the 'reviewBody' property of that part script 
            match_review = re.search(r'"reviewBody":"([^"]*)"', page_source)
            if match_review:
                review_body_text = match_review.group(1)
                data.loc[i, 'review_text'] = review_body_text
                print(f"{row[3]} and index = {i} SCRAPED")
            else:
                # there are approx. 500 web pages that were designed in a different way.
                soup = BeautifulSoup(page_source, 'html.parser')
                # Find the element that contains the main text
                main_text_element = soup.find('div', class_='contents')
                # Extract the main text
                main_text = main_text_element.get_text()
                data.loc[i,'review_text'] = main_text
                print(f"{row[3]} and index = {i} SCRAPED")
                
        except Exception as e:
            print(f"Exception {e}")
            data.loc[i, 'review_text'] = f"{e}"

            driver.quit()
            driver = webdriver.Edge("web drivers/msedgedriver.exe")
            driver.implicitly_wait(time_out_secs)
            driver.set_page_load_timeout(time_out_secs)
            continue
    
    data.to_csv(file_path,index = False)
    driver.quit()
