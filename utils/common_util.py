# Column descriptions
COLUMN_DESCRIPTIONS = {
    'id': 'Unique identifier for each house',
    'date': 'Date when the house was sold',
    'price': 'Sale price of the house (target variable)',
    'bedrooms': 'Total number of bedrooms',
    'bathrooms': 'Total number of bathrooms',
    'sqft_living': 'Interior living area in square feet',
    'sqft_lot': 'Lot size in square feet',
    'floors': 'Number of floors',
    'waterfront': 'Waterfront access (0 = No, 1 = Yes)',
    'view': 'View rating (0–4)',
    'condition': 'Condition rating (1–5)',
    'grade': 'Construction grade (1–13)',
    'sqft_above': 'Square footage above ground',
    'sqft_basement': 'Basement square footage',
    'yr_built': 'Year built',
    'yr_renovated': 'Year renovated',
    'zipcode': 'ZIP code',
    'lat': 'Latitude',
    'long': 'Longitude',
    'sqft_living15': 'Avg living area of 15 nearest houses',
    'sqft_lot15': 'Avg lot size of 15 nearest houses'
}

def get_all_cols_descrition():
    return COLUMN_DESCRIPTIONS

def get_col_description(col_name):
    return COLUMN_DESCRIPTIONS.get(col_name, "Column not found")

def get_lowercase(df):        
        df.columns= (df.columns.str.strip().str.lower().str.replace(" ", "_"))
        return df.columns

def get_target_col_last(df, target_col="price"):
    if target_col in df.columns:
        col = df.pop(target_col)
        df[target_col] = col
        return df.head()
    else:
         return f"Target column not found"
    
def split_dataset(df, target_col, test_size = 0.2, random_state= 42):
    
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("Random State:", random_state)
    print(f"Test size: {test_size*100}%")
    print("\nX_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    return X_train, X_test, y_train, y_test
