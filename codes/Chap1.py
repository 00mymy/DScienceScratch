# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:16:50 2017

@author: 00mymy
"""



users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

for user in users:
    user["friends"] = []

#확인하기
'''
for user in users:
    print(user)
    
users
users[0]
users['id'==0]
'''


for i, j in friendships:
    # this works because users[i] is the user whose id is i
    users[i]["friends"].append(users[j]) # add i as a friend of j
    users[j]["friends"].append(users[i]) # add j as a friend of i

#확인하기
'''
for user in users:
    print(user)
    
users
users[0]
'''

#확인해보니... 좀 더럽다.
#데이터 중복 줄여서 간단하게 만들기
'''
for user in users:
    user["friends"] = []
    
for i, j in friendships:
    users[i]["friends"].append(j) # add i as a friend of j
    users[j]["friends"].append(i) # add j as a friend of i
'''
  

def number_of_friends(user):
    """how many friends does _user_ have?"""
    return len(user["friends"])  		# length of friend_ids list

total_connections = sum(number_of_friends(user) for user in users)
num_users = len(users)
avg_connections = total_connections / num_users


# create a list (user_id, number_of_friends)
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]
num_friends_sorted = sorted(num_friends_by_id, key=lambda num_friends: num_friends[1], reverse=True) # largest to smallest

def friends_of_friend_ids_bad(user):
    # "foaf" is short for "friend of a friend"
    return [foaf["id"]
        for friend in user["friends"] # for each of user's friends
            for foaf in friend["friends"]] # get each of _their_ friends
            
# 데이터 중복을 없앤 수정된 자료구조 사용시
'''
def friends_of_friend_ids_bad(user):
    # "foaf" is short for "friend of a friend"
    return [foaf
        for friend in user["friends"] # for each of user's friends
            for foaf in users[friend]["friends"]] # get each of _their_ friends   
'''

friends_of_friend_ids_bad(users[0])

print([friend["id"] for friend in users[0]["friends"]]) # [1, 2]
print([friend["id"] for friend in users[1]["friends"]]) # [0, 2, 3]
print([friend["id"] for friend in users[2]["friends"]]) # [0, 1, 3]
# 데이터 중복을 없앤 수정된 자료구조 사용시
'''
print([friend for friend in users[0]["friends"]])
print([friend for friend in users[1]["friends"]])
print([friend for friend in users[2]["friends"]])
'''



from collections import Counter 	# not loaded by default

def not_the_same(user, other_user):
    """two users are not the same if they have different ids"""
    return user["id"] != other_user["id"]

def not_friends(user, other_user):
    """other_user is not a friend if he's not in user["friends"];
    that is, if he's not_the_same as all the people in user["friends"]"""
    return all(not_the_same(friend, other_user) for friend in user["friends"])

def friends_of_friend_ids(user):
    return Counter(foaf["id"]
    for friend in user["friends"] 		# for each of my friends
        for foaf in friend["friends"] 	# count *their* friends
        if not_the_same(user, foaf) 	# who aren't me
        and not_friends(user, foaf))    # and aren't my friends

'''
def friends_of_friend_ids(user):
    return Counter(foaf["id"]
    for friend in user["friends"] 		# for each of my friends
        for foaf in friend["friends"] 	# count *their* friends
        if not_the_same(user, foaf) 	# who aren't me
        and not_friends(user, foaf))    # and aren't my friends
'''

# 먼저 한 것과 비교해 보자
print(friends_of_friend_ids(users[0]))


# Friends-of-friend Look-up dict 만들기
from collections import defaultdict

fof_by_user = defaultdict(list)
for user in users:
    fof_by_user[user['id']].append(friends_of_friend_ids(user).most_common())


fof_by_user[3]






interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest):
    return [user_id
        for user_id, user_interest in interests
        if user_interest == target_interest]

#data_scientists_who_like('Big Data')
#[0, 8, 9]

from collections import defaultdict
# keys are interests, values are lists of user_ids with that interest
user_ids_by_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# keys are user_ids, values are lists of interests for that user_id
interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

#user_ids_by_interest['Big Data']
#[0, 8, 9]

#interests_by_user_id[0]
#['Hadoop', 'Big Data', 'HBase', 'Java', 'Spark', 'Storm', 'Cassandra']

#interests[0]
#(0, 'Hadoop')

def most_common_interests_with(user):
    return Counter(interested_user_id
        for interest in interests_by_user_id[user["id"]]
            for interested_user_id in user_ids_by_interest[interest]
                if interested_user_id != user["id"])





salaries_and_tenures = [
        (83000, 8.7), 
        (88000, 8.1),
        (48000, 0.7),
        (76000, 6),
        (69000, 6.5),
        (76000, 7.5),
        (60000, 2.5),
        (83000, 10),
        (48000, 1.9),
        (63000, 4.2)
]


#Scatter Plot
'''
import matplotlib.pyplot as plt
x = [ year for sal, year in salaries_and_tenures]
y = [ sal for sal, year in salaries_and_tenures]
plt.scatter(x,y)
'''

# keys are years, values are lists of the salaries for each tenure
salary_by_tenure = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# keys are years, each value is average salary for that tenure
average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
        for tenure, salaries in salary_by_tenure.items()
}

def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"
        
# keys are tenure buckets, values are lists of salaries for that bucket
salary_by_tenure_bucket = defaultdict(list)
for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)
    
# keys are tenure buckets, values are average salary for that bucket
average_salary_by_bucket = {
    tenure_bucket : sum(salaries) / len(salaries)
        for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}





def predict_paid_or_unpaid(years_experience):
    if years_experience < 3.0:
        return "paid"
    elif years_experience < 8.5:
        return "unpaid"
    else:
        return "paid"
        



words_and_counts = Counter(word
    for user, interest in interests
        for word in interest.lower().split())
        
for word, count in words_and_counts.most_common():
    if count > 1:
        print(word, count)
        
words_and_counts.most_common(5)
