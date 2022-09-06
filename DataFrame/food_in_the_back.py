from DataFrame.dataFrame import myDataFrame
from DataFrame.Altered_DataFrame import Altered_DataFrame
from Directories import home
import pandas as pd

myDataFrame = Altered_DataFrame(myDataFrame)
solver = 'ant'

# myDataFrame.df.to_excel('food_in_back.xlsx')
food_DataFrame = pd.read_excel(home + '\\DataFrame\\food_in_back.xlsx')

with_food = list(food_DataFrame['filename'][~(food_DataFrame['food in back'].str.strip() == 'no')])

myDataFrame.df['food in back'] = myDataFrame.df['filename'].isin(with_food)

DEBUG = 1