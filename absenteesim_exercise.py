import pandas as pd

raw_csv_data = pd.read_csv('data/Absenteeism-data.csv')
df = raw_csv_data.copy()

# dropping the ID columns
df = df.drop(['ID'], axis=1)
# print(df.head())

# converting the categorical column reason for absence into dummy columns
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
age_dummies = pd.get_dummies(df['Age'], drop_first=True)

# dividing the reason for absence column data into multiple dataset functionaly
reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
df = df.drop(['Reason for Absence'], axis=1)
df = pd.concat([df, reason_type_1, reason_type_2,
                reason_type_3, reason_type_4], axis=1)

column_names = ['Date',    'Transportation Expense',
                'Distance to Work',                       'Age',
                'Daily Work Load Average',           'Body Mass Index',
                'Education',                  'Children',
                'Pets', 'Absenteeism Time in Hours',
                'Reason_0',                           'Reason_1',
                'Reason_2',                           'Reason_3']
df.columns = column_names
#df_concatenated = pd.concat([df_no_age, age_dummies], axis=1)
# df_concatenated

column_names_reordered = ['Reason_0',                           'Reason_1',
                          'Reason_2',                           'Reason_3',
                          'Date',    'Transportation Expense',
                          'Distance to Work',                       'Age',
                          'Daily Work Load Average',           'Body Mass Index',
                          'Education',                  'Children',
                          'Pets', 'Absenteeism Time in Hours'
                          ]
df = df[column_names_reordered]

# df_concatenated.columns.values
# column_names = ['Reason for Absence', 'Date', 'Transportation Expense',
#        'Distance to Work', 'Daily Work Load Average', 'Body Mass Index',
#        'Education', 'Children', 'Pets', 27,
#        28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 46, 47, 48,
#        49, 50, 58, 'Absenteeism Time in Hours']
# df_concatenated = df_concatenated[column_names]
# df_concatenated

