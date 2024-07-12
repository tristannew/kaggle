import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/train.csv', index_col="id")

train, test = train_test_split(data, test_size=0.2, random_state=42)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.Region_Code = df.Region_Code.astype("int8")

    df.Previously_Insured = df.Previously_Insured.astype("int8")

    df.Driving_License = df.Driving_License.astype("int8")

    df.Gender = df.Gender.map({"Male": 1, "Female": 0})

    df.Vehicle_Damage = df.Vehicle_Damage.map({"Yes": 1, "No": 0})

    vehicle_age = pd.get_dummies(df.Vehicle_Age, drop_first=True)
    df[["< 1 Year", "> 2 Years"]] = vehicle_age
    df.drop(columns="Vehicle_Age", inplace=True)

    # region_dummies = pd.get_dummies(df.Region_Code, drop_first=True)
    # regions = pd.get_dummies(df.Region_Code, drop_first=True)
    # df.drop(columns="Region_Code", inplace=True)
    # df = pd.concat([df, regions], axis=1)
    return df

train = clean_data(train)
test = clean_data(test)