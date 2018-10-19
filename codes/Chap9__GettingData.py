# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 18:42:11 2017

@author: 00mymy
"""

import re
from collections import Counter


# Current Working Directory 확인
import os
cwd = os.getcwd()  # Get the current working directory (cwd)
# os.chdir("..")
# cwd = os.getcwd()
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))


#
# File Read
#

#os.chdir('Data_Science_from_Scratch')

hash_cnt = 0
def_cnt = 0
with open('Chap10.py','r') as f:
	for line in f: 				# look at each line in the file
         if re.match(r'^#',line): 		# use a regex to see if it starts with '#'
             hash_cnt += 1 	# if it does, add 1 to the count
         if re.match(r'^def [a-zA-Z0-9_(),=\s]+:$',line): 		# use a regex to see if it ends with '#'
             def_cnt += 1 	# if it does, add 1 to the count

print(hash_cnt)
print(def_cnt)



#
# Split & Strip
#
def get_domain(email_address):
    """split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]

with open('email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
                        for line in f if "@" in line)
    print(domain_counts)


my_str = "0000000this is string example....wow!!!0000000"
print(my_str.strip( '0' ))



#
#   CSV Read/Write
#                            
import csv
#with open('tab_delimited_stock_prices.txt', 'rt', encoding='utf-8') as f:
with open('tab_delimited_stock_prices.txt', 'rt') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        #print(row)
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        print(date, symbol, closing_price)

with open('tab_delimited_stock_prices.txt', 'rt') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        print(row)
        (date, symbol, closing_price) = row
        print(date, symbol, closing_price)



# 첫줄 무시 reader.__next__()
with open('colon_delimited_stock_prices.txt', 'rt') as f:
    reader = csv.reader(f, delimiter=':')
    reader.__next__()
    for row in reader:
        #print(row)
        (date, symbol, closing_price) = row
        print(date, symbol, closing_price)


# DictReader (첫 줄은 Key로 인식)
with open('colon_delimited_stock_prices.txt', 'rt') as f:
    reader = csv.DictReader(f, delimiter=':')
    for row in reader:
        #print(row)
        date = row["date"]
        symbol = row["symbol"]
        closing_price = float(row["closing_price"])
        print(date, symbol, closing_price)

# DictReader (Headerline 없을 때)        
with open('tab_delimited_stock_prices.txt', 'r') as f:
    reader = csv.DictReader(f, ('date', 'symbol', 'closing_price'), delimiter='\t')
    for row in reader:
        #print(row)
        date = row["date"]
        symbol = row["symbol"]
        closing_price = float(row["closing_price"])
        print(date, symbol, closing_price)


# Write
today_prices = { 'AAPL' : 92.92, 'MSFT' : 41.68, 'FB' : 64.5 }
#with open('comma_delimited_stock_prices.txt','wt', newline='') as f:
with open('comma_delimited_stock_prices.txt','wt') as f:
#with open('comma_delimited_stock_prices.txt','wt') as f:
    writer = csv.writer(f, delimiter=',')
    for stock, price in today_prices.items():
        writer.writerow([stock, price])
        

with open('comma_delimited_stock_prices.txt', 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        print(row)
    
    
    
#
# Web Scraping
#
from bs4 import BeautifulSoup
import requests
response = requests.get("http://www.yonsei.ac.kr")
html=response.text
# html[:1000]
html = " ".join(html.split())
# html[:1000]

#soup = BeautifulSoup(html, 'html5lib')
soup = BeautifulSoup(html, 'lxml')

#참고
'''
response.url
response.cookies
response.headers
response.status_code
'''

soup.find('p')
soup.find_all('p')
soup.p
soup.p.text
soup.p.text.split()
soup.p['id']        #Error
soup.p.get('id')    #None

all_p = soup.find_all('p')
all_p[:2]
p_with_ids = [p for p in soup.find_all('p') if p.get('class')]
p_with_ids = [p for p in soup('ul') if p.get('id')]

soup('div')
all_div = soup.find_all('div')
all_div[:2]
div_with_ids = [d for d in all_div if d.get('id')]
div_ids = [d.get('id') for d in all_div if d.get('id')]
div_classes = [d.get('class') for d in all_div if d.get('class')]

family = soup('div', {'class' : 'familysites'})
#soup.find_all("div", class_="familysites")
family2 = soup('div', 'familysites')
family3 = [d for d in soup('div') if 'familysites' in d.get('class', [])]


spans_in_divs = [span for div in soup('div') # for each <div> on the page
                            for span in div('span')]



# 여기부터는 인터넷이 되어야 실행/실습 가능
#############################################################################################
# O’Reilly Books About Data
#
# 세상은 그리 아름답지 않다...
#############################################################################################
url = "https://www.safaribooksonline.com/search/?query=data" + \
        "&extended_publisher_data=true&highlight=true&is_academic_institution_account=false&source=user" + \
        "&include_assessments=false&include_case_studies=true&include_courses=true&include_orioles=true" + \
        "&include_playlists=true&sort=publication_date"


# 단순하게 안됨. (비교 : 브라우저에서 소스보기 vs. 개발자도구에서 Element 보기)
'''
response = requests.get(url)
html=response.text
# html[:1000]
html = " ".join(html.split())
#html[:1000]
'''

# conda install selenium
from selenium import webdriver
from time import sleep

# Get Chromedriver at https://sites.google.com/a/chromium.org/chromedriver/home

# for headless mode
#options = webdriver.ChromeOptions()
#options.add_argument('headless')
#options.add_argument('window-size=1920x1080')
#options.add_argument("disable-gpu")


#driver = webdriver.Chrome('d:/Work/python/chromedriver.exe', options=options)
driver = webdriver.Chrome('d:/Work/python/chromedriver.exe')

driver.get(url)

sleep(3)
html = driver.page_source
#driver.quit()

soup = BeautifulSoup(html, 'lxml')
#soup = BeautifulSoup(html, 'html5lib')

book_list = soup.find_all('div', {'class':'meta'})

def book_info(book):
    title = book('div','book-title')[0].a.text.strip()
    authors = [author['title'] for author in book('span', 'author')[0]('a')]
    isbn_link = book('div', 'book-title')[0].a['href']
    isbn = re.match('/library/view/(.*)', isbn_link).groups()[0].split('/')[1]
    #publisher = book('span', 'publisher t-publisher')[0].a['title']
    date = book('span','publish-date')[0].text
    return {
            "title" : title,
            "authors" : authors,
            "isbn" : isbn,
            #"publisher" : publisher
            "date" : date
}

for book in book_list:
    b_info = book_info(book)
    print( b_info)



#
# 
from selenium import webdriver
from time import sleep

driver = webdriver.Chrome('d:/Work/python/chromedriver.exe')


base_url = "https://www.safaribooksonline.com/search/?query=data" + \
        "&extended_publisher_data=true&highlight=true&is_academic_institution_account=false&source=user" + \
        "&include_assessments=false&include_case_studies=true&include_courses=true&include_orioles=true" + \
        "&include_playlists=true&sort=publication_date"


books = []
PAGE_SIZE = 10
num_result = int(soup.find('span', {'class':'js-results-total'}).text)
NUM_PAGES = round(num_result/PAGE_SIZE)


#for page_num in range(1, NUM_PAGES + 1):
for page_num in range(1, 3):
    print( "souping page", page_num, ",", len(books), " found so far" )
    url = base_url + '&page='+ str(page_num)
    driver.get(url)
    sleep(5)
    html = driver.page_source

    soup = BeautifulSoup(html, 'lxml')
    #soup = BeautifulSoup(html, 'html5lib')
    
    book_list = soup.find_all('div', {'class':'meta'})
    for book in book_list:
        books.append(book_info(book))
    # now be a good citizen and respect the robots.txt!
    sleep(30)


'''    
# 직접 해보자....
def get_year(book):
    """book["date"] looks like 'November 2014' so we need to
    split on the space and then take the second piece"""
    return int(book["date"].split()[2])


year_counts = Counter(get_year(book) for book in books
                      if get_year(book) <= 2019)

import matplotlib.pyplot as plt
years = sorted(year_counts)
book_counts = [year_counts[year] for year in years]
plt.plot(years, book_counts)
plt.ylabel("# of data books")
plt.title("Data is Big!")
plt.show()
'''


#############################################################################################
# Using API
#
# 
#############################################################################################


#handling JSON
import json
serialized = """{ "title" : "Data Science Book",
    "author" : "Joel Grus",
    "publicationYear" : 2014,
    "topics" : [ "data", "science", "data science"] }"""
# parse the JSON to create a Python dict
deserialized = json.loads(serialized)
if "data science" in deserialized["topics"]:
    print(deserialized)
    
#비교
type(serialized)
type(deserialized)



# GiyHub
endpoint = "https://api.github.com/users/joelgrus/repos"
repos = json.loads(requests.get(endpoint).text)


# Twitter
from twython import Twython
CONSUMER_KEY = 'WBRg9QIYXYL3s2vh00TyABa52'
CONSUMER_SECRET = '0DV8UyuvI4PGTJgTGUdUH2wwj4IKqlzPGs2hD27rH2Mhy2lH4U'
twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET)

# search for tweets containing the phrase "data science"
for status in twitter.search(q='"손흥민"')["statuses"]:
    user = status["user"]["screen_name"]#.encode('utf-8')
    text = status["text"]#.encode('utf-8')
    print(user, ":", text)


from twython import TwythonStreamer
tweets = []

class MyStreamer(TwythonStreamer):  
    def on_success(self, data):
        # only want to collect English-language tweets
        if data['lang'] == 'en':
            tweets.append(data)
            print("received tweet # : ", len(tweets))
        # stop when we've collected enough
        if len(tweets) >= 5:
            self.disconnect()
    def on_error(self, status_code, data):
        print(status_code, data)
        self.disconnect()


ACCESS_TOKEN = '29871883-OupFSTWDYVHVkkO34XtkR5O66imlaDYyCa2cA0OXT'
ACCESS_TOKEN_SECRET = 'Ev559KAz11cc8wSN13TEXliXGftJtzqwhbufyTBYY1qlt'

stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# starts consuming public statuses that contain the keyword 'data'
stream.statuses.filter(track='world cup')

top_hashtags = Counter(hashtag['text'].lower()
                        for tweet in tweets
                            for hashtag in tweet["entities"]["hashtags"])
print(top_hashtags.most_common(5))
