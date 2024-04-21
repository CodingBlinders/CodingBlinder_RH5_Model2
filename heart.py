import pandas as pd
import random
random.seed(1)

#read in the dataset (select 2015)
year = '2015'
brfss_2015_dataset = pd.read_csv(f'./{year}.csv')

#How many rows and columns
brfss_2015_dataset.shape

#check that the data loaded in is in the correct format
pd.set_option('display.max_columns', 500)
brfss_2015_dataset.head()

"""**At this point we have 441,456 records and 330 columns. Each record contains an individual's BRFSS survey responses.**"""

# select specific columns
brfss_df_selected = brfss_2015_dataset[['_MICHD',
                                         '_RFHYPE5',
                                         'TOLDHI2', '_CHOLCHK',
                                         '_BMI5',
                                         'SMOKE100',
                                         'CVDSTRK3', 'DIABETE3',
                                         '_TOTINDA',
                                         '_FRTLT1', '_VEGLT1',
                                         '_RFDRHV5',
                                         'HLTHPLN1', 'MEDCOST',
                                         'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK',
                                         'SEX', '_AGEG5YR', 'EDUCA', 'INCOME2' ]]

brfss_df_selected.shape

brfss_df_selected.head()

"""## 2. Clean the data

### 2.1 Drop missing values
"""

#Drop Missing Values - knocks 100,000 rows out right away
brfss_df_selected = brfss_df_selected.dropna()
brfss_df_selected.shape

"""### 2.2 Modify and clean the values to be more suitable to ML algorithms
In order to do this part, I referenced the codebook which says what each column/feature/question is: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf
"""

# _MICHD
#Change 2 to 0 because this means did not have MI or CHD
brfss_df_selected['_MICHD'] = brfss_df_selected['_MICHD'].replace({2: 0})
brfss_df_selected._MICHD.unique()

#1 _RFHYPE5
#Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
brfss_df_selected['_RFHYPE5'] = brfss_df_selected['_RFHYPE5'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFHYPE5 != 9]
brfss_df_selected._RFHYPE5.unique()

#2 TOLDHI2
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['TOLDHI2'] = brfss_df_selected['TOLDHI2'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI2 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI2 != 9]
brfss_df_selected.TOLDHI2.unique()

#3 _CHOLCHK
# Change 3 to 0 and 2 to 0 for Not checked cholesterol in past 5 years
# Remove 9
brfss_df_selected['_CHOLCHK'] = brfss_df_selected['_CHOLCHK'].replace({3:0,2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._CHOLCHK != 9]
brfss_df_selected._CHOLCHK.unique()

#4 _BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
brfss_df_selected['_BMI5'] = brfss_df_selected['_BMI5'].div(100).round(0)
brfss_df_selected._BMI5.unique()

#5 SMOKE100
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['SMOKE100'] = brfss_df_selected['SMOKE100'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 9]
brfss_df_selected.SMOKE100.unique()

#6 CVDSTRK3
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['CVDSTRK3'] = brfss_df_selected['CVDSTRK3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 9]
brfss_df_selected.CVDSTRK3.unique()

#7 DIABETE3
# going to make this ordinal. 0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['DIABETE3'] = brfss_df_selected['DIABETE3'].replace({2:0, 3:0, 1:2, 4:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE3 != 9]
brfss_df_selected.DIABETE3.unique()

#8 _TOTINDA
# 1 for physical activity
# change 2 to 0 for no physical activity
# Remove all 9 (don't know/refused)
brfss_df_selected['_TOTINDA'] = brfss_df_selected['_TOTINDA'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._TOTINDA != 9]
brfss_df_selected._TOTINDA.unique()

#9 _FRTLT1
# Change 2 to 0. this means no fruit consumed per day. 1 will mean consumed 1 or more pieces of fruit per day
# remove all dont knows and missing 9
brfss_df_selected['_FRTLT1'] = brfss_df_selected['_FRTLT1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._FRTLT1 != 9]
brfss_df_selected._FRTLT1.unique()

#10 _VEGLT1
# Change 2 to 0. this means no vegetables consumed per day. 1 will mean consumed 1 or more pieces of vegetable per day
# remove all dont knows and missing 9
brfss_df_selected['_VEGLT1'] = brfss_df_selected['_VEGLT1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._VEGLT1 != 9]
brfss_df_selected._VEGLT1.unique()

#11 _RFDRHV5
# Change 1 to 0 (1 was no for heavy drinking). change all 2 to 1 (2 was yes for heavy drinking)
# remove all dont knows and missing 9
brfss_df_selected['_RFDRHV5'] = brfss_df_selected['_RFDRHV5'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFDRHV5 != 9]
brfss_df_selected._RFDRHV5.unique()

#12 HLTHPLN1
# 1 is yes, change 2 to 0 because it is No health care access
# remove 7 and 9 for don't know or refused
brfss_df_selected['HLTHPLN1'] = brfss_df_selected['HLTHPLN1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.HLTHPLN1 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.HLTHPLN1 != 9]
brfss_df_selected.HLTHPLN1.unique()

#13 MEDCOST
# Change 2 to 0 for no, 1 is already yes
# remove 7 for don/t know and 9 for refused
brfss_df_selected['MEDCOST'] = brfss_df_selected['MEDCOST'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST != 9]
brfss_df_selected.MEDCOST.unique()

#14 GENHLTH
# This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
# Remove 7 and 9 for don't know and refused
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 9]
brfss_df_selected.GENHLTH.unique()

#15 MENTHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['MENTHLTH'] = brfss_df_selected['MENTHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 99]
brfss_df_selected.MENTHLTH.unique()

#16 PHYSHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['PHYSHLTH'] = brfss_df_selected['PHYSHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 99]
brfss_df_selected.PHYSHLTH.unique()

#17 DIFFWALK
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
brfss_df_selected['DIFFWALK'] = brfss_df_selected['DIFFWALK'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 9]
brfss_df_selected.DIFFWALK.unique()

#18 SEX
# in other words - is respondent male (somewhat arbitrarily chose this change because men are at higher risk for heart disease)
# change 2 to 0 (female as 0). Male is 1
brfss_df_selected['SEX'] = brfss_df_selected['SEX'].replace({2:0})
brfss_df_selected.SEX.unique()

#19 _AGEG5YR
# already ordinal. 1 is 18-24 all the way up to 13 wis 80 and older. 5 year increments.
# remove 14 because it is don't know or missing
brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 14]
brfss_df_selected._AGEG5YR.unique()

#20 EDUCA
# This is already an ordinal variable with 1 being never attended school or kindergarten only up to 6 being college 4 years or more
# Scale here is 1-6
# Remove 9 for refused:
brfss_df_selected = brfss_df_selected[brfss_df_selected.EDUCA != 9]
brfss_df_selected.EDUCA.unique()

#21 INCOME2
# Variable is already ordinal with 1 being less than $10,000 all the way up to 8 being $75,000 or more
# Remove 77 and 99 for don't know and refused
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME2 != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME2 != 99]
brfss_df_selected.INCOME2.unique()

#Check the shape of the dataset now: We have 253,680 cleaned rows and 22 columns (1 of which is our dependent variable)
brfss_df_selected.shape

#Let's see what the data looks like after Modifying Values
brfss_df_selected.head()

#Check Class Sizes of the heart disease column
brfss_df_selected.groupby(['_MICHD']).size()

from sklearn.model_selection import train_test_split

X = brfss_df_selected.drop('_MICHD', axis=1)  # Features
y = brfss_df_selected['_MICHD']  # Target variable

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Step 3: Choose a machine learning algorithm
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Train the model using the training data
clf.fit(X_train, y_train)

# Step 5: Evaluate the model using the testing data
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print(classification_report(y_test, y_pred))

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Define the FastAPI app
app = FastAPI()

# Define the request body model
class HeartPredictionRequest(BaseModel):
    RFHYPE5: int
    TOLDHI2: int
    CHOLCHK: int
    BMI5: int
    SMOKE100: int
    CVDSTRK3: int
    DIABETE3: int
    TOTINDA: int
    FRTLT1: int
    VEGLT1: int
    RFDRHV5: int
    HLTHPLN1: int
    MEDCOST: int
    GENHLTH: int
    MENTHLTH: int
    PHYSHLTH: int
    DIFFWALK: int
    SEX: int
    AGEG5YR: int
    EDUCA: int
    INCOME2: int

# Define column mapping for the new structure
column_mapping = {
    "RFHYPE5": "_RFHYPE5",
    "TOLDHI2": "TOLDHI2",
    "CHOLCHK": "_CHOLCHK",
    "BMI5": "_BMI5",
    "SMOKE100": "SMOKE100",
    "CVDSTRK3": "CVDSTRK3",
    "DIABETE3": "DIABETE3",
    "TOTINDA": "_TOTINDA",
    "FRTLT1": "_FRTLT1",
    "VEGLT1": "_VEGLT1",
    "RFDRHV5": "_RFDRHV5",
    "HLTHPLN1": "HLTHPLN1",
    "MEDCOST": "MEDCOST",
    "GENHLTH": "GENHLTH",
    "MENTHLTH": "MENTHLTH",
    "PHYSHLTH": "PHYSHLTH",
    "DIFFWALK": "DIFFWALK",
    "SEX": "SEX",
    "AGEG5YR": "_AGEG5YR",
    "EDUCA": "EDUCA",
    "INCOME2": "INCOME2"
}

# Define the endpoint
@app.post("/predict_heart_disease")
async def predict_heart_disease(request_data: HeartPredictionRequest):
    # Convert request data to dictionary
    new_data = request_data.dict()

    # Map keys to new structure
    new_data_value = {column_mapping[key]: value for key, value in new_data.items()}

    # # Create a DataFrame
    new_data_df = pd.DataFrame(new_data_value, index=[0])

    # # Use the trained model to predict whether the individual is a heart patient or not
    prediction = clf.predict(new_data_df)

    # Define response message
    if prediction[0] == 1:
        prediction_message = "The individual is predicted to be a heart patient."
    else:
        prediction_message = "The individual is predicted not to be a heart patient."

    # Return response as JSON
    return {"prediction": prediction_message}
