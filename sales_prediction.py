# ***Step-A: Data Preprocessing:-***

print("\n" + "-"*50)
print("📌 Step A: Data Preprocessing - Preparing the Data 📌")
print("-"*50)

# region -  Step 1: Importing Required Libraries

print("🚀 Step 1: Importing Required Libraries")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import nltk
import warnings
warnings.filterwarnings('ignore')
# Set the option to prevent silent downcasting
np.set_printoptions(formatter={'float': '{:,.2f}'.format}, suppress=True, precision=2)
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('future.no_silent_downcasting', True)
print("\t ✅ Required Libraries are Imported - Done!\n")
# endregion

# region - Step-2: Load the Dataset and Display Different overviews of Datasets:-

print("📂 Step-2: Load the Dataset and Display Different overviews of Datasets:- ")
""" ## """
dataset = pd.DataFrame(pd.read_csv('D:/AI ENGINEERING/Datasets/ML/Big_mart_sales/train_data.csv'))
print("\t ✅ Dataset Loaded Successfully!\n")

# Seperate the whole datset into categorical and numerical columns...
categorical_cols = dataset.select_dtypes(include=['object']).columns
numerical_cols = dataset.select_dtypes(exclude=['object']).columns

if categorical_cols.empty:
  print("\t No Categorical Columns Found...\n")
else:
  print("\t ✅ Categorical Columns Found: ", categorical_cols, "\n")

if numerical_cols.empty:
  print("\t No Numerical Columns Found...\n")
else:
  print("\t ✅ Numerical Columns Found: ", numerical_cols, "\n")

print("> Shape of the Dataset:", dataset.shape, "\n")
print("\n> Information about Dataset:", dataset.info())
print("\n> Statistical summary of the Dataset: \n", dataset.describe().map(lambda x: round(x, 4)), "\n")

# plt.figure(figsize=(10, 6))
# sns.distplot(dataset.iloc[:, -1], bins=30, kde=True)
# plt.title('Distribution of Last Column')
# plt.show()

# # Plot target distribution
# sns.histplot(dataset.iloc[:, -1], bins=30, kde=True)
# plt.title('Distribution of Last Column')
# plt.show()

# # Correlation Matrix Heatmap
# sns.heatmap(dataset[numerical_cols].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix \n',fontsize=20,  fontweight=800)
# plt.show()

# # Boxplot for OutletType vs OutletSales
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='OutletType', y='OutletSales', data=dataset)
# plt.xticks(rotation=45)
# plt.title('OutletType vs OutletSales')
# plt.show()

#endregion

# region - Step-3:- Checking the Dataset:-

print("⏳️ Step 3: Checking and Processing the Dataset:-")
print("🔎 Step 3.1: Checking for Duplicates:-")
if dataset.duplicated().any():
  dataset.drop_duplicates(inplace=True)
  print("\t ✅ Duplicate Data(or Identical Rows) found and Removed...\n")
else:
    print("\t No Duplicate Data(or Identical Rows) found...\n")

print("🔎 Step-3.2: Checking any Missing Data:-")
# Here from the module named impute of the library scikit-learn, we are using the SimpleImputer Class to Handle the Missing Values.
from sklearn.impute import SimpleImputer

missing_data_counts = dataset.isnull().sum() + dataset.isin(['', 'N/A', 'Unknown', 'NaN']).sum()

if missing_data_counts.any():
  categorical_missing_counts = dataset[categorical_cols].isnull().sum() + dataset[categorical_cols].isin(['', 'N/A', 'Unknown', 'NaN']).sum()
  numerical_missing_counts = dataset[numerical_cols].isnull().sum()

  # Replace "Unknown" with NaN in categorical columns
  for col in categorical_cols:
    dataset[col] = dataset[col].replace('Unknown', np.nan)

  # Check if there are any missing values (categorical or numerical)
  if categorical_missing_counts.any() or numerical_missing_counts.any():
      # Print missing counts for categorical columns in the desired format
      print("\t ⚠️ Missing Data Found! Filling missing values...")
      # Create imputers for categorical and numerical features
      categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
      numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

      # Apply imputers to the selected columns in X
      if len(categorical_cols) > 0:
        categorical_cols_for_impution = [col for col in categorical_cols if col != dataset.columns[-1]]
        dataset[categorical_cols_for_impution] = categorical_imputer.fit_transform(dataset[categorical_cols_for_impution])

      # Exclude the dependent variable column (last column) if it's numerical
      numerical_cols_for_impution = [col for col in numerical_cols if col != dataset.columns[-1]]
      if len(numerical_cols_for_impution) > 0:
          dataset[numerical_cols_for_impution] = numerical_imputer.fit_transform(dataset[numerical_cols_for_impution])
      # if len(numerical_cols) > 0:
          # dataset[numerical_cols] = numerical_imputer.fit_transform(dataset[numerical_cols])
      print("\t ✅ Missing Data Handled Successfully!\n")
else:
    print("\t No missing data found...\n")

print("🔎 Step-3.3: Checking any Synonyms or Aliases:-")
for col in dataset.columns:
  # Get value counts and convert to DataFrame with column name
  # .reset_index(name='Counts: '): This converts the Series into a DataFrame and names the count column as 'Counts: '.
  value_counts_dataset = dataset[col].value_counts().rename_axis('Unique Values: ').reset_index(name='Counts: ')

  # Transpose and print with formatting
  # print(f"Column: \t'{col}'")
  # .to_string(header=False): This converts the transposed DataFrame to a string for printing and removes the header row.
  # print(value_counts_dataset.T.to_string(header=False), "\n")

print("\t ⚠️ Some Aliases or Synonyms found! Handling them...\n")
# # Handling ProductID Column
dataset['ProductID'] = dataset['ProductID'].apply(lambda x: x[:2])
# dataset['ProductID'] = dataset['ProductID'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})
print("\t ✅ After Handling the Prefixes of 'ProductID' Column and changed to new Name: {'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'}")
# print(dataset['ProductID'].value_counts())

# Handling FatContent Column
dataset['FatContent'] = dataset['FatContent'].replace({ 'LF': 'Low Fat', 'low fat': 'Low Fat', 'Low fat': 'Low Fat', 'reg': 'Regular'})
dataset.loc[dataset['ProductID'] == 'NC', 'FatContent'] = 'Non-Edible'
print("\t ✅ After Handling the Aliases of 'FatContent' Column: Low Fat and Regular & Adding a new Category: Non-Edible")
# print(dataset['FatContent'].value_counts())

# Handling EstablishmentYear Column
import datetime as dt
current_year = dt.datetime.today().year
dataset['OutletAge'] = current_year - dataset['EstablishmentYear']
dataset = dataset.drop('EstablishmentYear', axis=1)
print("\t ✅ After Handling the Aliases of 'EstablishmentYear' column is Deleted, Instead 'OutletAge' column is created...\n")
# print(dataset['OutletAge'].value_counts())

print("🔎 Step-3.4: Checking for Stopwords:-")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
if categorical_cols.any():
  nltk.download('stopwords')
  stop_words = set(stopwords.words('english'))
  print("\t ⚠️ Stopwords found! Stemming Needed...")
  def stemming(text):
      words = text.lower().split()

      stemmer = PorterStemmer()
      stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]
      return ' '.join(stemmed_words)  # Join stemmed words back into a string

  # Apply the stemming function to the specified columns
  for column in categorical_cols:
      dataset[column] = dataset[column].astype(str).apply(stemming)

  print("\t ✅ Stemming Completed... \n")
  # print(dataset.head().to_string(header=True))
else:
  print("\t No Stemming Needed...\n")

print("🔎 Step-3.5: Checking any Categorical Data:-")
import scipy.sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1].values

repeating_cols = []
for col in categorical_cols:
    # Matrix of Feature
    if col != dataset.columns[-1]:
      # print(f"> String Values present in Column '{col}'.")
      # Check for repeating values within the categorical column
      value_counts = dataset[col].value_counts()
      repeating_values = value_counts[value_counts > 1].index.tolist()
      if repeating_values:
        repeating_cols.append(col)
        print(f"\t Categorical values found in column '{col}': {repeating_values}.")
        print(f"\t - '{col}' is Encoded Successfully...\n")
      else:
        print(f"\t But No Categorical values found in column '{col}'.\n")

    # Dependent Variable/Output
    if col == dataset.columns[-1]:
      print(f"> String Values present in Column '{col}'(Output Column).")
      # Check for repeating values within the categorical column
      value_counts = dataset[col].value_counts()
      repeating_values = value_counts[value_counts > 1].index.tolist()
      if repeating_values:
        print(f"\t Categorical values found in column '{col}': {repeating_values}")
        le = LabelEncoder()
        y = le.fit_transform(dataset[col])
        print(f"\t -'{col}' is Encoded Successfully...\n")
      else:
        print(f"\t But No Categorical values found in column '{col}'.\n")

# print("=> Repeating Columns in Matrix of Features(X): ", repeating_cols, "\n")
if repeating_cols:
  encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), repeating_cols)], remainder='passthrough')
  encoder_transform = encoder.fit_transform(X)

  # Conditional conversion to dense array
  if scipy.sparse.issparse(encoder_transform):  # Check if sparse
    X = encoder_transform.toarray()
  else:
    X = encoder_transform

else:
  print("\t No Repeating Columns found in Matrix of Features(X)... \n")
print("\t ✅ Categorical values Encoded Successfully!...\n ")

# print("Matrix of Features(X): \n", pd.DataFrame(X).head().to_string(header=True))
# print("\n")
# print("Dependent Variable(y): \n", pd.DataFrame(y).head().to_string(header=False, index=False))
# endregion

# region - Step-4: Split the Dataset into the Training set and Test set:-
print (" ✂️ Step 4: Split Dataset into the Training set and Test set:-")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Printing Training Sets: ")
# print("> X_train: \n", X_train)
# print("> X_test: \n", X_test, "\n")
# print("\n")
# print("Printing Test Sets: ")
# print("> y_train: \n", y_train)
# print("> y_test: \n", y_test)
print("\t ✅ Splitting Completed Successfully...\n  ")
# endregion

# region - Step-5: Feature Scaling:-"""
print ("⚖️ Step 5: Feature Scaling:-")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Convert X_train and X_test to NumPy arrays if they are DataFrames
X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
# Iterate through columns of X_train and X_test
for col in range(X_train.shape[1]):  # Use range to get column indices
    # Check if all values in the column are 0 or 1
    if np.all(np.isin(X_train[:, col], [0, 1])):
        continue  # Skip scaling for this column
    else:
        # Reshape the column before scaling
        X_train[:, col] = scaler.fit_transform(X_train[:, col].reshape(-1, 1)).flatten()
        X_test[:, col] = scaler.transform(X_test[:, col].reshape(-1, 1)).flatten()

# print("Printing Training Sets after Feature Scaling:")
# print("> X_train: \n", X_train)
# print(pd.DataFrame(X).head().to_string())
# print("\n")
# print("Printing Test Sets after Feature Scaling:")
# print("> X_test: \n", X_test)
print("\t ✅ Feature Scaling Completed Successfully...\n")
#endregion

print("\n" + "-"*50)
print("📌 Step B: Model Training & Evaluation 📌")
print("-"*50)

# region - Step-1: Model Training:-
print("📚 Step-1: Model Training:-")
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print('\t ✅ Model - "Random Forest Regressor" has been Trained Successfully...\n')
#endregion

# region - Step-2: Model Evaluation:-
print("💡 Step-2: Model Evaluation:-")
y_pred = model.predict(X_test)
cv_score = cross_val_score(model, X, y, cv=5)
print(f"\t -> {model.__class__.__name__}:-")
print(f"\t R2 Score: \t{r2_score(y_test, y_pred):.2f}")
print(f"\t CV Score : \t{cv_score.mean()*100:.2f}%")
print(f"\t MSE: \t\t{mean_squared_error(y_test, y_pred):.2f}")
print(f"\t RMSE : \t{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"\t MAE : \t\t{mean_absolute_error(y_test, y_pred):.2f}\n")
# endregion

# region - Step-3: Model Testing:-
print("\t Printing Sample Data & Output Sales to test the model:-")
print("\t Sample Data: ", pd.DataFrame(X_train).iloc[0].values.reshape(1, -1), "\n")
print("\t Output Sales: ",pd.DataFrame(y_train).iloc[0], "\n")
# endregion

print("\n" + "-"*50)
print("📌 Step C: Saving the Model 📌")
print("-"*50)

# region - Saving the Model:-
import pickle
with open("./saved_model/sales_prediction.sav", "wb") as model_file:
    pickle.dump(model, model_file)

model = pickle.load(open("./saved_model/sales_prediction.sav", "rb"))
print("\t ✅ Model saved successfully...\n")

sample_input = pd.DataFrame(X_train).iloc[0].values.reshape(1, -1)  # Convert a row to an array
prediction = model.predict(sample_input)
print("\t > Predicted Sales:", prediction[0])
# endregion

print("\n" + "="*50)
if prediction[0] == (pd.DataFrame(y_train).iloc[0].values):
    print("🏁🎉 Hurray! your Prediction is correct... 🎉🏁")
else:
    print(" 🙁 Oops! your Prediction is incorrect... 🙁")
