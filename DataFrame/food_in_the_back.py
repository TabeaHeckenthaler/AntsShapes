from DataFrame.dataFrame import myDataFrame
from DataFrame.Altered_DataFrame import Altered_DataFrame
from Directories import home
from DataFrame.import_excel_dfs import df_ant
import pandas as pd

# myDataFrame = Altered_DataFrame(myDataFrame)
solver = 'ant'

# myDataFrame.df.to_excel('food_in_back.xlsx')
food_DataFrame.to_excel('food_in_back.xlsx')
food_DataFrame = pd.read_excel(home + '\\DataFrame\\food_in_back.xlsx')
# add to the food_DataFrame the rows from df_ant whos filename are not in food_DataFrame
for filename in df_ant['filename']:
    if filename not in food_DataFrame['filename'].values:
        food_DataFrame = food_DataFrame.append(df_ant[df_ant['filename'] == filename])

food_DataFrame['filename']=df_ant['filename']
DEBUG = 1


with_food = list(food_DataFrame['filename'][~(food_DataFrame['food in back'].str.strip() == 'no')])

myDataFrame.df['food in back'] = myDataFrame.df['filename'].isin(with_food)

DEBUG = 1