# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:21:06 2018

@author: 00mymy
"""

#survived,pclass,name,sex,age
titanic = [
    [1,1,"Smith, Mrs. Lucien Philip","F",18],
    [0,3,"Gustafsson, Mr. Karl Gideon","M",19],
    [1,3,"Jalsevac, Mr. Ivan","M",29],
    [0,2,"Hiltunen, Miss. Marta","F",18],
    [0,2,"Fillbrook, Mr. Joseph Charles","M",18],
    [1,3,"McCoy, Miss. Alicia","F",25],
    [0,3,"Lindahl, Miss. Agda Thorilda Viktoria","F",25],
    [0,2,"Butler, Mr. Reginald Fenton","M",25],
    [0,3,"Harknett, Miss. Alice Phoebe","F",32],
    [0,3,"O'Brien, Mr. Timothy","M",49],
    [1,1,"Wilson, Miss. Helen Alice","F",31],
    [0,3,"Sage, Mr. Frederick","M",38],
    [0,3,"Palsson, Miss. Stina Viola","F",33],
    [1,3,"O'Brien, Mrs. Thomas","F",37],
    [1,3,"Persson, Mr. Ernst Ulrik","M",25],
    [1,1,"Hays, Mrs. Charles Melville","F",52],
    [0,3,"Vander Planke, Miss. Augusta Maria","F",18],
    [0,3,"Brobeck, Mr. Karl Rudolf","M",22],
    [1,1,"Salomon, Mr. Abraham L","M",63],
    [0,1,"Chisholm, Mr. Roderick Robert Crispin","M",51],
    [0,3,"Leinonen, Mr. Antti Gustaf","M",32],
]




# 3등석 탑승자 중에서 생존자는 몇 명인가?
c3 = [p for p in titanic if p[1]==3]
c3_sur = [p for p in c3 if p[0]==1]
print("3등석 생존자 수 :", len(c3_sur))
print("-------------------------------\n\n")


# 탑승자들의 평균 나이, 최고령자 나이, 최연소자 나이를 남자/여자 각각 구하시오.
m_ages = [p[4] for p in titanic if p[3]=='M']
f_ages = [p[4] for p in titanic if p[3]=='F']
print("(최연소)", "남:", min(m_ages), "여:", min(f_ages))
print("(최고령)", "남:", max(m_ages), "여:", max(f_ages))
print("(평균연령)", "남:", sum(m_ages)/len(m_ages), "여:", sum(f_ages)/len(f_ages))





# 생존자 목록을 나이순(내림차)으로 출력하시오.
survived = [p for p in titanic if p[0]==1]
survived = sorted(survived, key=lambda p: p[4], reverse=True)

print("생존자 목록 - 나이순")
for s in survived:
    print(s)

print("-------------------------------\n\n")



# Class(객실등급)별 생존자수를 막대 그래프로 그리시오 (X축:클래스, Y축:생존자수)

from collections import Counter
from matplotlib import pyplot as plt

classes = [p[1] for p in titanic if p[0]==1]
class_count = Counter(classes)

x_values, y_values = [], []
for c, n in sorted(class_count.most_common()):
    x_values.append(c)
    y_values.append(n)


plt.bar(x_values, y_values)
plt.xlabel("Passenger Classes")
plt.ylabel("# of Passengers")
plt.title("Titanic Passengers by Classes")
plt.xticks([i+1 for i, _ in enumerate(x_values)], x_values)
plt.show()


print("-------------------------------\n\n")

# 전체 탑승자, 생존자, 비생존자 각각에 대해 나이대(10대, 20대, ...) 분포를 막대 그래프로 그리시오. (X축:나이대, Y축:탑승자수)
def draw_ages(ages, title):
    decile = lambda age: age // 10 * 10
    histogram = Counter(decile(age) for age in ages)
    #plt.bar([x - 4 for x in histogram.keys()], histogram.values(), 8)
    plt.bar([x for x in histogram.keys()], histogram.values(), 8)
    #plt.axis([-5, 100, 0, 50]) 	# x-axis -5 to 105,  y-axis 0 to 5
    plt.xticks([10 * i for i in range(11)]) 	
    plt.xlabel("Age Groups")
    plt.ylabel("# of Passengers")
    plt.title(title)
    plt.show()

all_ages = [p[4] for p in titanic]
survied_ages = [p[4] for p in titanic if p[0]==1]
not_survied_ages = [p[4] for p in titanic if p[0]==0]
draw_ages(survied_ages, "Titanic Passengers by Ages - Survived")
draw_ages(not_survied_ages, "Titanic Passengers by Ages - Not Survived")
draw_ages(all_ages, "Titanic Passengers by Ages - All")


print("-------------------------------\n\n")
# List로 표현된 전체 탑승자 정보를 Dictionary 형식으로 변경
from collections import defaultdict
passengers = defaultdict(dict)
 
for p in titanic:
    p_name = p[2].split(',')[0]
    p_data = {'survived':p[0], 'pclass':p[1], 'name':p[2], 'sex':p[3], 'age':p[4]}
    passengers[p_name] = p_data
    
print(passengers)
