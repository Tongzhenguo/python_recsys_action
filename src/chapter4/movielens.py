# encoding: utf-8


"""


@author: tongzhenguo


@time: 2018/6/26 下午10:27


@desc:ml-20m数据集初步分析
"""

import pandas as pd

# 设置显示宽度,避免控制台显示折行
pd.set_option('display.width', 1000)

# 查看评分数据
ratings = pd.read_csv("../../data/ml-20m/ratings.csv")
print(ratings.head())
"""
   userId  movieId  rating   timestamp
0       1        2     3.5  1112486027
1       1       29     3.5  1112484676
2       1       32     3.5  1112484819
3       1       47     3.5  1112484727
4       1       50     3.5  1112484580
"""

# 查看电影数据
movies = pd.read_csv("../../data/ml-20m/movies.csv")
print(movies.head())
"""
   movieId                               title                                       genres
0        1                    Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy
1        2                      Jumanji (1995)                   Adventure|Children|Fantasy
2        3             Grumpier Old Men (1995)                               Comedy|Romance
3        4            Waiting to Exhale (1995)                         Comedy|Drama|Romance
4        5  Father of the Bride Part II (1995)                                       Comedy
"""

# 查看标签数据
tags = pd.read_csv("../../data/ml-20m/tags.csv", encoding='utf-8')
print(tags.head())
"""
   userId  movieId            tag   timestamp
0      18     4141    Mark Waters  1240597180
1      65      208      dark hero  1368150078
2      65      353      dark hero  1368150079
3      65      521  noir thriller  1368149983
4      65      592      dark hero  1368150078
"""

# 拼接评分数据和电影数据
df = pd.merge(ratings, movies, on="movieId")
df.to_pickle("../../cache/merge_ratings_movies.pkl")
print(df.head())
"""
   userId  movieId  rating   timestamp           title                      genres
0       1        2     3.5  1112486027  Jumanji (1995)  Adventure|Children|Fantasy
1       5        2     3.0   851527569  Jumanji (1995)  Adventure|Children|Fantasy
2      13        2     3.0   849082742  Jumanji (1995)  Adventure|Children|Fantasy
3      29        2     3.0   835562174  Jumanji (1995)  Adventure|Children|Fantasy
4      34        2     3.0   846509384  Jumanji (1995)  Adventure|Children|Fantasy
"""

# 将timestamp转换为YYYYMMDD和weekday的形式
import arrow

df["YYYYMMDD"] = df["timestamp"].apply(lambda ts: arrow.get(ts).format("YYYYMMDD"))
df["weekday"] = df["timestamp"].apply(lambda ts: arrow.get(ts).format("dddd"))
df.to_pickle("../../cache/merge_ratings_movies.pkl")
print(df.head())
"""
   userId  movieId  rating   timestamp           title                      genres  YYYYMMDD    weekday
0       1        2     3.5  1112486027  Jumanji (1995)  Adventure|Children|Fantasy  20050402   Saturday
1       5        2     3.0   851527569  Jumanji (1995)  Adventure|Children|Fantasy  19961225  Wednesday
2      13        2     3.0   849082742  Jumanji (1995)  Adventure|Children|Fantasy  19961127  Wednesday
3      29        2     3.0   835562174  Jumanji (1995)  Adventure|Children|Fantasy  19960623     Sunday
4      34        2     3.0   846509384  Jumanji (1995)  Adventure|Children|Fantasy  19961028     Monday
"""

# matplotlib柱状图显示一周行为数据
df = pd.read_pickle("../../cache/merge_ratings_movies.pkl")
week_top10 = df[df['YYYYMMDD'].isin(map(str, range(19960623, 19960630)))][["rating", "movieId"]] \
    .groupby(["movieId"]) \
    .sum().reset_index().sort_values(by="rating", ascending=False).head(10)
print(week_top10.head())
"""
     movieId   rating
343      380  15800.0
478      595  15726.0
327      364  15420.0
320      356  14744.0
472      588  14650.0
"""

import matplotlib.pyplot as plt

name_list = week_top10["movieId"].values
num_list = week_top10["rating"].values
plt.bar(range(len(num_list)), num_list, tick_label=name_list)
plt.show()

df = pd.read_pickle("../../cache/merge_ratings_movies.pkl")
df = df['genres'].str.split('|', expand=True).stack().reset_index(level=0).set_index('level_0').rename(
    columns={0: 'genre'}) \
    .join(df.drop('genres', axis=1))
df.to_pickle("../../cache/merge_ratings_movies.pkl")

# matplotlib饼状图显示各内容分类占比
genre_distr = df[["genre", "movieId"]].groupby("genre").count().reset_index()
print(genre_distr)
"""
                  genre  movieId
0   (no genres listed)      361
1               Action  5614208
2            Adventure  4380351
3            Animation  1140476
4             Children  1669249
5               Comedy  7502234
6                Crime  3298335
7          Documentary   244619
8                Drama  8857853
9              Fantasy  2111403
10           Film-Noir   216689
11              Horror  1482737
12                IMAX   492366
13             Musical   870915
14             Mystery  1557282
15             Romance  3802002
16              Sci-Fi  3150141
17            Thriller  5313506
18                 War  1048618
19             Western   423714
"""

from matplotlib import pyplot as plt
genre_distr = genre_distr.sample(frac=1.0)
labels = genre_distr["genre"].values
weights = list(genre_distr["movieId"])
plt.pie(weights,            # 角度权重
        labels=labels,      # 各个扇区显示的标识
        labeldistance=1.2,  # label到扇区的距离
        autopct='%2.0f%%',  # 扇区内显示的数值,比如10%
        startangle=90,      # 第一个扇区对应的水平角度
        pctdistance=0.85,   # 扇区数值到饼图中心的距离
        center=[-5.0, 0])   # 饼图中心坐标
plt.axis('equal')  # 官方建议,为了饼状图更美观
plt.legend()
plt.show()