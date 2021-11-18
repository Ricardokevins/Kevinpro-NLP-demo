# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:24:58 2018
@author: ZZH
"""




import requests
import re
import time
import pandas as pd
from bs4 import BeautifulSoup

from collections import Counter
import os
import random




def get_one_page(url):
    proxies = {'http': None, 'https': None}
    response = requests.get(url, proxies=proxies)
    #print(response.status_code) 
    while response.status_code == 403:
        time.sleep(500 + random.uniform(0, 500))  
        response = requests.get(url, proxies=proxies)
        #print(response.status_code)
    #print(response.status_code)
    if response.status_code == 200:
        return response.text

    return None


def stripOutDay(date_string):
    date_string = str(date_string).split(',')[1]
    date_string = date_string.split(' ')[1]
    return date_string

import datetime

def main():
    
    today = time.strftime("%Y-%m-%d")
    yesterday = datetime.datetime.now() - datetime.timedelta(days = 0)
    yesterday_day = yesterday.strftime("%d")
    yesterday = yesterday.strftime("%Y-%m-%d")

    if not os.path.exists(yesterday):
        os.makedirs(yesterday)
    url = 'https://arxiv.org/list/cs/pastweek?show=1000'
    html = get_one_page(url)
    soup = BeautifulSoup(html, features='html.parser')
    content = soup.dl
    date = soup.find('h3')
    ul_list = soup.find_all(name='h3')
    contents = soup.find_all(name='dl')
    
    search_yesterday_content = None
    #print()
    if(yesterday_day[0] == '0'):
        yesterday_day = yesterday_day[1:]
    for index,i in zip(ul_list,contents):
        print(stripOutDay(index),yesterday_day)
        if(stripOutDay(index) == yesterday_day):
            search_yesterday_content = i


    if (search_yesterday_content == None):
        print("++++++++++++++++++ hit bug +++++++++++++++++")
        exit()
    
    list_ids = search_yesterday_content.find_all('a', title = 'Abstract')
    list_title = search_yesterday_content.find_all('div', class_ = 'list-title mathjax')
    list_authors = search_yesterday_content.find_all('div', class_ = 'list-authors')
    list_subjects = search_yesterday_content.find_all('div', class_ = 'list-subjects')
    list_subject_split = []
    for subjects in list_subjects:
        subjects = subjects.text.split(': ', maxsplit=1)[1]
        subjects = subjects.replace('\n\n', '')
        subjects = subjects.replace('\n', '')
        subject_split = subjects.split('; ')
        list_subject_split.append(subject_split)

    items = []
    for i, paper in enumerate(zip(list_ids, list_title, list_authors, list_subjects, list_subject_split)):
        if 'Computation and Language (cs.CL)' in paper[4]:
            items.append([paper[0].text, paper[1].text, paper[2].text, paper[3].text, paper[4]])
    name = ['id', 'title', 'authors', 'subjects', 'subject_split']
    paper = pd.DataFrame(columns=name,data=items)
    paper.to_csv(yesterday + "/" + yesterday+'_'+str(len(items))+'.csv')


    '''subject split'''
    subject_all = []
    for subject_split in list_subject_split:
        for subject in subject_split:
            subject_all.append(subject)
    subject_cnt = Counter(subject_all)
    #print(subject_cnt)
    subject_items = []
    for subject_name, times in subject_cnt.items():
        subject_items.append([subject_name, times])
    subject_items = sorted(subject_items, key=lambda subject_items: subject_items[1], reverse=True)
    name = ['name', 'times']
    subject_file = pd.DataFrame(columns=name,data=subject_items)
    #subject_file = pd.DataFrame.from_dict(subject_cnt, orient='index')
    #subject_file.to_csv(time.strftime("%Y-%m-%d")+'_'+str(len(items))+'.csv')
    #subject_file.to_html('subject_file.html')    
    # count = 0 
    # for index,row in paper.iterrows():
    #     if('Computation and Language (cs.CL)' in row['subject_split']  ):
    #         print(index,row['subject_split'], row['title'],type(row['subject_split'])) 
    #         count += 1

    # print(count)

    
    '''key_word1 selection'''
    Key_words = ['factual', 'dialogue summarization', 'summary', 'summarization', 'faithful','translation','Contrastive Learning']
    selected_papers = paper[paper['title'].str.contains(Key_words[0], case=False)]
    for key_word in Key_words[1:]:
        selected_paper1 = paper[paper['title'].str.contains(key_word, case=False)]
        selected_papers = pd.concat([selected_papers, selected_paper1], axis=0)
    
    selected_papers.to_csv(yesterday + "/" + yesterday+'_'+str(len(selected_papers))+'.csv')
    content = 'Today arxiv has {} new papers in CS area, and {} of them is about NLP, {} of them contain your keywords.\n\n'.format(len(list_title), subject_cnt['Computation and Language (cs.CL)'], len(selected_papers))
    content += 'Ensure your keywords is ' + str(Key_words)  + '(case=True). \n\n'
    content += 'This is your paperlist.Enjoy! \n\n'
    print(content)
    for i, selected_paper in enumerate(zip(selected_papers['id'], selected_papers['title'], selected_papers['authors'], selected_papers['subject_split'])):
        #print(content1)
        content1, content2, content3, content4 = selected_paper
        content += '------------' + str(i+1) + '------------\n' + content1 + content2 + str(content4) + '\n'
        content1 = content1.split(':', maxsplit=1)[1]
        content += 'https://arxiv.org/abs/' + content1 + '\n\n'



    content += 'Here is the Research Direction Distribution Report. \n\n'
    for subject_name, times in subject_items:
        content += subject_name + '   ' + str(times) +'\n'
    title = yesterday + ' you have {} papers'.format(len(selected_papers))
    #send_email(title, content)
    freport = open(yesterday + "/" + 'title'+'.txt', 'w')
    freport.write(content)
    freport.close()

    #exit()
    '''dowdload key_word selected papers'''
    list_subject_split = []
    
    for selected_paper_id, selected_paper_title in zip(selected_papers['id'], selected_papers['title']):
        selected_paper_id = selected_paper_id.split(':', maxsplit=1)[1]
        selected_paper_title = selected_paper_title.split(':', maxsplit=1)[1]
        proxies = {'http': None, 'https': None}
        url = 'https://arxiv.org/pdf/' + selected_paper_id
        r = requests.get(url, proxies=proxies)
        
        while r.status_code == 403:
            time.sleep(100 + random.uniform(0, 500))
            r = requests.get('https://arxiv.org/pdf/' + selected_paper_id,proxies=proxies)
        selected_paper_id = selected_paper_id.replace(".", "_")
        pdfname = selected_paper_title.replace("/", "_")   #pdf名中不能出现/和：
        pdfname = pdfname.replace("?", "_")
        pdfname = pdfname.replace("\"", "_")
        pdfname = pdfname.replace("*","_")
        pdfname = pdfname.replace(":","_")
        pdfname = pdfname.replace("\n","")
        pdfname = pdfname.replace("\r","")
        print('%s %s'%(selected_paper_id.strip(), selected_paper_title.strip()))
        with open(yesterday+'/%s %s.pdf'%(selected_paper_id,pdfname), "wb") as code:    
           code.write(r.content)



if __name__ == '__main__':
    main()
    time.sleep(1)