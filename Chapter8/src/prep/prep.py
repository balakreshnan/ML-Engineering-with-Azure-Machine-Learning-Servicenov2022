

import argparse
import pandas as pd

parser = argparse.ArgumentParser("prep")
parser.add_argument("--raw_data", type=str, help="Path to raw data")
parser.add_argument("--prep_data", type=str, help="Path of prepped data")
args = parser.parse_args()

print(args.raw_data)
print(args.prep_data)


df = pd.read_csv(args.raw_data)

df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))
df['Sex']= df['Sex'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
df['Loc']= df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
df['Embarked'] = df['Embarked'].fillna('S')
df.loc[:,'GroupSize'] = 1 + df['SibSp'] + df['Parch']

LABEL = 'Survived'
columns_to_keep = ['Pclass', 'Sex','Age', 'Fare', 'Embared', 'Deck', 'GroupSize']
columns_to_drop = ['Name','SibSp', 'Parch', 'Survived']
df_train = df
df = df_train.drop(['Name','SibSp', 'Parch', 'PassengerId'], axis=1)

df.to_csv(args.prep_data)
