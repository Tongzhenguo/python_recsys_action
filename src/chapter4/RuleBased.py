# encoding: utf-8


"""
@author: tongzhenguo


@time: 2018/6/26 上午8:04


@desc:使用movielens数据集演练基于规则的召回策略


"""

# 引入相关类
from pyspark.sql import SparkSession
import sys
import arrow
import pyspark
from pyspark.sql.types import StringType, DoubleType

# 初始化SparkSession实例
APP_NAME = "RuleBased"
spark = SparkSession.builder.appName(APP_NAME).getOrCreate()

# pyspark加载movielens数据集
proj_path = '/'.join(sys.path[0].split('/')[:-2])
ratings = spark.read.csv(r'file://%s/data/ml-20m/ratings.csv' % proj_path, header=True)
movies = spark.read.csv(r'file://%s/data/ml-20m/movies.csv' % proj_path, header=True)
df = ratings.join(movies, on="movieId", how="inner")
df.createOrReplaceTempView("temp_raw_table")


# 定义函数get_weekday,其根据输入的时间戳转换为星期几的字符串
def get_weekday(timestamp):
    """
    根据给定的时间字符串,返回对应的星期几
    :param timestamp:unix时间戳
    :return: str,例如"sunday"
    """
    return arrow.get(timestamp).format("dddd")


# 注册timestamp2weekday为hive的udf
sqlContext = pyspark.HiveContext(spark.sparkContext)
sqlContext.registerFunction('timestamp2weekday', get_weekday, StringType())

raw_table_query = """
select yyyy_mm_dd
    ,weekday
    ,timestamp
    ,userId
    ,movieId
    ,title
    ,genre
    ,rating
from 
(
    select split(from_unixtime(timestamp),' ')[0] as yyyy_mm_dd
        ,timestamp2weekday(timestamp) as weekday
        ,timestamp
        ,title
        ,genres
        ,userId
        ,movieId
        ,rating
    from temp_raw_table
) a
lateral view explode(split(a.genres,'\\\|')) as genre
"""

# 显示处理的结果,并注册为临时表raw_table
df = spark.sql(raw_table_query)
df.createOrReplaceTempView("raw_table")
df.cache()
# df.show()
"""
+----------+--------+------+-------+--------------------+---------+------+
|yyyy_mm_dd| weekday|userId|movieId|               title|    genre|rating|
+----------+--------+------+-------+--------------------+---------+------+
|2005-04-03|Saturday|     1|      2|      Jumanji (1995)|Adventure|   3.5|
|2005-04-03|Saturday|     1|      2|      Jumanji (1995)| Children|   3.5|
|2005-04-03|Saturday|     1|      2|      Jumanji (1995)|  Fantasy|   3.5|
|2005-04-03|Saturday|     1|     29|City of Lost Chil...|Adventure|   3.5|
|2005-04-03|Saturday|     1|     29|City of Lost Chil...|    Drama|   3.5|
|2005-04-03|Saturday|     1|     29|City of Lost Chil...|  Fantasy|   3.5|
|2005-04-03|Saturday|     1|     29|City of Lost Chil...|  Mystery|   3.5|
|2005-04-03|Saturday|     1|     29|City of Lost Chil...|   Sci-Fi|   3.5|
|2005-04-03|Saturday|     1|     32|Twelve Monkeys (a...|  Mystery|   3.5|
|2005-04-03|Saturday|     1|     32|Twelve Monkeys (a...|   Sci-Fi|   3.5|
|2005-04-03|Saturday|     1|     32|Twelve Monkeys (a...| Thriller|   3.5|
|2005-04-03|Saturday|     1|     47|Seven (a.k.a. Se7...|  Mystery|   3.5|
|2005-04-03|Saturday|     1|     47|Seven (a.k.a. Se7...| Thriller|   3.5|
|2005-04-03|Saturday|     1|     50|Usual Suspects, T...|    Crime|   3.5|
|2005-04-03|Saturday|     1|     50|Usual Suspects, T...|  Mystery|   3.5|
|2005-04-03|Saturday|     1|     50|Usual Suspects, T...| Thriller|   3.5|
|2004-09-10|  Friday|     1|    112|Rumble in the Bro...|   Action|   3.5|
|2004-09-10|  Friday|     1|    112|Rumble in the Bro...|Adventure|   3.5|
|2004-09-10|  Friday|     1|    112|Rumble in the Bro...|   Comedy|   3.5|
|2004-09-10|  Friday|     1|    112|Rumble in the Bro...|    Crime|   3.5|
+----------+--------+------+-------+--------------------+---------+------+
"""

# 计算最近一周中评分最高的top500
latest_week_query = "select yyyy_mm_dd from raw_table group by yyyy_mm_dd order by yyyy_mm_dd desc limit 7"
latest_week = spark.sql(latest_week_query).rdd.map(lambda row: row[0]).collect()
print(latest_week)
"""
[u'2015-03-31', u'2015-03-30', u'2015-03-29', u'2015-03-28', u'2015-03-27', u'2015-03-26', u'2015-03-25']
"""

# 规则一:计算最近一周评分最高的500个电影id
top500_rating_query = """
select movieId
    ,sum(rating) as week_rating
from raw_table
where yyyy_mm_dd in ({latest_week})
group by movieId
order by week_rating desc
limit 500
""".format(latest_week=','.join(map(lambda x: "'%s'" % x,latest_week)))
df = spark.sql(top500_rating_query)
print(top500_rating_query)
"""
select movieId
    ,sum(rating) as week_rating
from raw_table
where yyyy_mm_dd in ('2015-03-31','2015-03-30','2015-03-29','2015-03-28','2015-03-27','2015-03-26','2015-03-25')
group by movieId
order by week_rating desc
limit 500
"""

df.createOrReplaceTempView("top500_ratings")
df.show()
"""
+-------+-----------+
|movieId|week_rating|
+-------+-----------+
|  79132|     2548.0|
|  58559|     1334.0|
|   7153|     1190.0|
|   2571|     1143.0|
|    356|     1122.0|
|    296|     1008.0|
|   2959|      998.0|
| 109487|      971.0|
| 112852|      970.5|
|   4306|      969.0|
|  60069|      935.0|
|  72998|      888.0|
|  78499|      780.0|
|  68954|      740.0|
|    318|      707.0|
|   6539|      704.0|
|   6016|      690.0|
|    260|      681.0|
|    593|      676.5|
|     50|      673.5|
+-------+-----------+
"""

# 规则二:评分小于等于1的占比不足5%
top500_ratings_filter_badrating_query = """
select a.movieId
    ,a.week_rating
from top500_ratings a
join (
    select movieId
        ,sum(case when rating<=1 then 1 else 0 end) as bad_rat_cnt
        ,count(rating) as rat_cnt
    from raw_table
    group by movieId
) b
on a.movieId = b.movieId 
where 100.0*b.bad_rat_cnt/b.rat_cnt<=5
order by week_rating desc 
"""
df = spark.sql(top500_ratings_filter_badrating_query)
df.createOrReplaceTempView("top500_ratings_filter_badrating")
print(df.count())  # 454
df.show()
"""
+-------+-----------+
|movieId|week_rating|
+-------+-----------+
|  79132|     2548.0|
|  58559|     1334.0|
|   7153|     1190.0|
|   2571|     1143.0|
|    356|     1122.0|
|    296|     1008.0|
|   2959|      998.0|
| 109487|      971.0|
| 112852|      970.5|
|   4306|      969.0|
|  60069|      935.0|
|  72998|      888.0|
|  78499|      780.0|
|  68954|      740.0|
|    318|      707.0|
|   6539|      704.0|
|   6016|      690.0|
|    260|      681.0|
|    593|      676.5|
|     50|      673.5|
+-------+-----------+
"""

# 规则三：近实时热榜--根据hacker news热门排序算法获取小时热榜


def calculate_score(votes, item_hour_age, gravity=1.8):
    """
    Hacker News的ranking算法
    :param votes:投票数
    :param item_hour_age:现在到提交经历的小时数
    :param gravity:时间重力因子
    :return:score
    """
    return (votes - 1) / pow((item_hour_age+2), gravity)


# 注册hive临时函数HackerNewsScore
sqlContext = pyspark.HiveContext(spark.sparkContext)
sqlContext.registerFunction('HackerNewsScore', calculate_score, DoubleType())


# 根据Hacker News的ranking算法筛选top100作为最近一小时的近实时热榜
top100_1h_query = """
select movieId
    ,HackerNewsScore(sum(rating),(max(timestamp)-min(timestamp))/3600) as HNScore
from temp_raw_table
where from_unixtime(timestamp) between '2015-03-31 09:00:00' and '2015-03-31 09:59:59'
group by movieId
order by HNScore desc
limit 100
"""
df = spark.sql(top100_1h_query)
df.show()
"""
+-------+------------------+
|movieId|           HNScore|
+-------+------------------+
|  96588| 1.148698354997035|
| 116857| 1.148698354997035|
|  68554| 1.148698354997035|
|   1305| 1.148698354997035|
|   4914| 1.148698354997035|
| 100390| 1.148698354997035|
| 109846| 1.148698354997035|
|   2432| 1.148698354997035|
|   3703|1.0051110606224056|
|  33072|1.0051110606224056|
|   3681|1.0051110606224056|
|   3160|1.0051110606224056|
|   2951|1.0051110606224056|
|   1199|1.0051110606224056|
| 110586|0.8615237662477763|
|   7894|0.8615237662477763|
|  25927|0.8615237662477763|
|   7438|0.8615237662477763|
|   1213|0.8615237662477763|
|   8914|0.8615237662477763|
+-------+------------------+
"""
