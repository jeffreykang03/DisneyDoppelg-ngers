import pandas as pd
import csv

df1 =  pd.read_csv('/Users/claire/Desktop/DF1.csv').drop(columns = ['personality_text'])
df2 =  pd.read_csv('/Users/claire/Desktop/DF2.csv').drop(columns = ['image', 'link', 'Unnamed: 0']).drop(9)

k_s = pd.read_csv('/Users/claire/Desktop/K_and_S.csv').drop(columns = ['image', 'link', 'Unnamed: 0'])


df2 = pd.concat([k_s, df2])

df3 = df2.merge(df1, left_on='character', right_on='character_name').drop(columns = ['character_name', 'Unnamed: 0'])
df3.rename(columns={'dicription': 'discription'}, inplace=True)

df1_char = [df1.get('character_name')]
df2_char = [df2.get('character')]

traits_words = df3.get('traits').str.replace("'", '')
traits_words = traits_words.str.strip('[]')
traits_words = traits_words.str.split(', ')

df3 = df3.assign(traits_words = traits_words)
df3 = df3.assign(traits_words = df3.get('traits_words').apply(lambda x: ", ".join(x[:5])))
df3 = df3.drop(columns = 'traits')
df3.rename(columns={'traits_words': 'traits'}, inplace=True)

df3.to_csv("/Users/claire/Desktop/DF3.csv", index= True)

print(df3.columns)