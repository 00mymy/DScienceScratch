# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:18:20 2017

@author: 00mymy
"""

from collections import defaultdict

scores = [
        {"name":"kim", "score" : [("kor", 100), ("math", 90), ("eng",80), ("dsc",100)]},
        {"name":"lee", "score" : [("kor", 50), ("math", 50), ("eng",50), ("dsc",100)]},
        {"name":"seo", "score" : [("kor", 90), ("math", 80), ("eng",90), ("dsc",100)]},
        # and more
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
  
def get_avg_by_student():
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
    for course in course_sum.keys():
        avg_by_course[course] = course_sum[course] / course_cnt[course]
    return avg_by_course

def get_student_scores_by_course():
    course_scores = defaultdict(list)
    for stud in scores:
        for score in stud['score']:
            course_scores[score[0]].append((stud['name'], score[1]))
    return course_scores

def get_best_student():
    std_score_list = list(get_sum_by_name().items())
    std_score_list.sort(key=lambda x:x[1])
    return std_score_list[-1]
    
def get_best_student_of_course():
    course_best = defaultdict(tuple)
    student_scores_by_course = get_student_scores_by_course()
    for c_name, c_scores in student_scores_by_course.items():
        c_scores.sort(key=lambda x:x[1])
        course_best[c_name] =c_scores[-1]
    return course_best
    
        
std_score_list = list(get_sum_by_name().items())
std_score_list.sort(key=lambda x:x[1])

#print("sum_by_name =", get_sum_by_name())
#print("sum_by_course =", get_sum_by_course())
print("Students Average =", dict(get_avg_by_student()))
print("Courses Average =", dict(get_avg_by_course()))
print("Best student =", get_best_student())
print("Best student by course=", dict(get_best_student_of_course()))



# Visualization
c_avg_dict = dict(get_avg_by_course())
c_avg = sorted(c_avg_dict.items(), key=lambda x:x[1], reverse=True)
k_list = [k for k, _ in c_avg]
v_list = [v for _, v in c_avg]

xs = [i + 0.1 for i, _ in enumerate(k_list)]

from matplotlib import pyplot as plt
plt.bar(xs, v_list)
plt.ylabel("Avg. Score")
plt.ylim(ymax = 100)
plt.title("Course Averages")
plt.xticks([i + 0.5 for i, _ in enumerate(k_list)], k_list)
plt.show()