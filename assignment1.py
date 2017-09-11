# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:18:20 2017

@author: 00mymy
"""

from collections import defaultdict

scores = [
    {"name":"kim", "score" : [("kor", 100), ("math", 90), ("eng",80)]},
    {"name":"lee", "score" : [("kor", 50), ("math", 50), ("eng",50)]},
    {"name":"seo", "score" : [("kor", 80), ("math", 90), ("eng",100)]}
    ]
    
    
def get_sum_by_name():
    by_name = defaultdict(int)
    for stud in scores:
        by_name[stud['name']]= sum([s for c, s in stud['score']])
    return by_name

def get_sum_by_course():
    by_course = defaultdict(int)
    for stud in scores:
        for score in stud['score']:
            by_course[score[0]] += score[1]
    return by_course    
  
def get_avg_by_name():
    by_name = defaultdict(int)
    for stud in scores:
        by_name[stud['name']]= sum([s for c, s in stud['score']]) / len(stud['score'])
    return by_name

def get_avg_by_course():
    course_sum = defaultdict(int)
    course_cnt = defaultdict(int)
    for stud in scores:
        for score in stud['score']:
            course_sum[score[0]] += score[1]
            course_cnt[score[0]] += 1
            
    avg_by_course = defaultdict(int)
    for stud in course_sum.keys():
        avg_by_course[stud] = course_sum[stud] / course_cnt[stud]
    return avg_by_course
    
print("sum_by_name =", get_sum_by_name())
print("sum_by_course =", get_sum_by_course())
print("avg_by_name =", get_avg_by_name())
print("avg_by_course =", get_avg_by_course())
# best student
# best student by course
