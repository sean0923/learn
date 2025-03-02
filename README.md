# Learn in Public

#### 2025-03-01-Sat

- still learning more about gmail api with python

#### 2025-02-28-Fri

- Learned chroma_db without langchain
- set up gmail read only api to fetch emails
  
#### 2025-02-27-Thu

- setup `celery` for job queue

#### 2025-02-26-Wed

- setup `pinecone` vector database

#### 2025-02-25-Tue

- learn about limit of `intermediate_steps` and how using `memory` will remember to keep track of some history between two different agent executors 

#### 2025-02-24-Mon

- `agent`: chain that knows how to use tools
- `agent executor`: runs agent until it is not a function call (fancy while loop)
- learn how to make tools
- seems like `chain` concept is important but not sure I am getting it...

#### 2025-02-23-Sun

- learn about calculate how similar vectors are (L2, Cosine similarity)
- use OpenAI Embeddings to change text -> embeddings -> save to chroma db (sqlite based vector db)
- use retriever chain to work with OpenAI chat with embeddings

#### 2025-02-22-Sat
- intro to LangChain
- learn LangChain has many utilities to help using LLM
- one chaine / using sequential chains to talk to openai llm
- there is concept called `Memeory` and using it was helpful to keep chat history to openai api

#### 2025-02-21-Fri
- hand written number recognition using multinomial-logistic-regression with gradient descent using `Tensorflow.js`

#### 2025-02-20-Thu
- logistic-regression with gradient descent using `Tensorflow.js`

#### 2025-02-19-Wed
- linear-regression with gradient descent using `Tensorflow.js`

#### 2025-02-18-Tue
- learning basic of linear-regression with gradient descent

#### 2025-02-17-Mon
- Modal.com (run ml python function in cloud)
- intro to tensorflowJs

#### 2025-02-16-Sun
- train yolo with sku-110k in colab, took 7 hours and result looked good for stright image

#### 2025-02-15-Sat
- learn more about gradio and implemented yolo object detection with gradio

#### 2025-02-14-Fri
- start predicting amazon item price using LLM. Dataset comes from: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

#### 2025-02-13-Thu
- intro to LangChain

#### 2025-02-12-Wed
- intro to RAG

#### 2025-02-11-Tue

- learn multiple LLM leader board websties
  - [vellum](https://www.vellum.ai/llm-leaderboard): has price info
  - [lmarena](https://lmarena.ai/?leaderboard): human ranking of LLM based on AB testing
  - [SEAL](https://scale.com/leaderboard): specific topic

#### 2025-02-10-Mon

- learn `gr.ChatInterface(fn=func)`
- learn `tools` that goes into one of the argument to `openai.chat.completions.create`

#### 2025-02-09-Sun

- learn cool tool called `gradio` (UI generation tool that can take function, input, output)
![image](https://github.com/user-attachments/assets/85665cf9-8355-43dd-b825-b66c813d0767)


#### 2025-02-08-Sat

- download ollama and ran deepseek-r1 locally for the first time
- ran llama3.3 70b (43GB) locally and shocked at how slow the response came back (about 1 word/min)

#### 2025-02-07-Fri

- save and load trained model

#### 2025-02-06-Thu

- trained deep learning model with about 68% accuracy on identifying dog breed with 800, 200 training and validation set

#### 2025-02-05-Wed

- Start building deep learning model using `keras`, [`google/mobilenet-v2`](https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/130-224-classification)

#### 2025-02-04-Tue

- Turn images into Tensors

#### 2025-02-03-Mon

- Start learning about deep learning, TensorFlow.
- Instead of using local Jupyter Notebook, using Google colab with GPU access
  
#### 2025-02-02-Sun

- Calculated score with training and validation set
- Decreased model.fit time setting `max_samples` much lower than actual data size
```py
# Change max_samples values to decrease training time
model = RandomForestRegressor(n_jobs=-1, random_state=42, max_samples=10000) # sample is about 400,000
```

#### 2025-02-01-Sat

- Fill missing numerical values then fill missing non-numerical values
- Fit through RandomForestRegressor (about 400,000 rows --> takes about 1 min to fit)

#### 2025-01-31-Fri

- covert object type to integer using pd `category`
```py
content.astype("category").cat.as_ordered() # string value will turn into integer
```

#### 2025-01-30-Thu

- it's good idea to sort by date when working with time series data 
```py
df_temp.sort_values(by=["saledate"], inplace=True, ascending=True)
```

#### 2025-01-29-Wed

- start project wiht time series data (bulldozer price prediction) (regression)

#### 2025-01-28-Tue

- evaluation with `sklearn.metrics`

#### 2025-01-27-Mon

- hyperparameter tuning with RandomSearchCV and GridSearchCV
 
#### 2025-01-26-Sun

- hyperparameter tuning for KNN
  
#### 2025-01-25-Sat

- correlation matrix `df.corr()`

#### 2025-01-24-Fri

- heart-disease-project (hdp): try to understand data by looking at 2 independant variable at the same time (age vs max heart rate)

#### 2025-01-23-Thu

- Started heart disease project
- Analyzing the data
  - how is the target distribution `df["target"].value_counts`
  - is there any missing values? `df.isna().sum()`
  - how is individual feature vs target looks like?

#### 2025-01-22-Wed

- Learned sklearn.pipline to go through car_sales with missing data (imputer to fill up na, encoder to convert to numeric using pipeline and preprocessor)

#### 2025-01-21-Tue

- save & load model with pickle and joblib

#### 2025-01-20-Mon

- Leraned sklearn `RandomizedSearchCV` and `GridSearchCV` for hyperparameter tuning

#### 2025-01-19-Sun

- Divide set into 3: train, validate, test, train, then apply hyperparameter tuning to validation set
- Hyperparmeter Tuning by hand in RandomForestClassifier model (`n_estimators`, `max_depth`).

#### 2025-01-18-Sat

- Learnd parameter vs hyperparameter concept. Will adjust hyperparameter to improve the model performance in the future

#### 2025-01-17-Fri

- Cross Validation Scoring for Classification and Regression

#### 2025-01-16-Thu

- Mean Square Error (MSE): amplify big diff btw y_test and y_pred

#### 2025-01-15-Wed
- Mean Absolute Error (MAE)
![image](https://github.com/user-attachments/assets/4c12df3c-ac2b-40f4-86f2-1b2ae25feac1)


#### 2025-01-14-Tue

- Learn intro to regression model evaluation metrics
  - R^2 (coefficient of determination) <-- only learned this
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

#### 2025-01-13-Mon

- Learn confusion matrix

#### 2025-01-12-Sun

- Learning regression & classfication model using `RandomForestRegressor` `RandomForestClassifier`
- Learn `predict_proba` vs `predict`
- `ROC curve` & `roc_auc_score`

#### 2025-01-11-Sat

- Scikit-learn using `RandomForestRegressor`
```py
# refit the model
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

#### 2025-01-10-Fri

- Scikit-learn intro

#### 2025-01-09-Thu

- more matplot lib (style & custom subplots)
- export plot as png
- matplot lib exercise

#### 2025-01-08-Wed

- plot from pandas data frame
![image](https://github.com/user-attachments/assets/6e7f7084-4374-40bc-98dd-636268497189)


#### 2025-01-07-Tue
- matplotlib histogray and subplots
![image](https://github.com/user-attachments/assets/2204b7bf-a279-4cb6-a644-7e98c03a7523)



#### 2025-01-06-Mon
- np.argsort() <-- sort index of array
- intro to matplotlib
- fig, ax = plt.subplots()
![image](https://github.com/user-attachments/assets/750530ae-8454-481d-aaab-91a5115d5835)


#### 2025-01-05-Sun

- intro to numpy
  - `np.array()`
```py
a3 = np.array([[[1,2,3],
               [4,5,6]],
               
              [[2,3,4],
               [2,3,4]],
               
              [[3,4,5],
               [3,4,5]]])
```
- `range_array = np.arange(3,7,2)`
- `randint_array = np.random.randint(1, 10, size = (2,3))`
```py
np.random.seed(seed=1) # setting seed make the output same every you run the code
random_array_1 = np.random.randint(10, size=(1,10))
```
- `%timeit` <-- jupyter notebook magic function
- python `sum()` vs `np.sum()` <-- np.sum() is much faster
- np.std()
- reshape and transpose
- dot product  
![image](https://github.com/user-attachments/assets/43d89ea6-4da7-49fc-af45-276ffc10e2b3)




#### 2025-01-04-Sat
- finish pandas assignment 1
![image](https://github.com/user-attachments/assets/13acb649-b1e0-413d-8ca2-99f4e884010e)


#### 2025-01-03-Fri
- start assignment 1 - pandas practice
![image](https://github.com/user-attachments/assets/ace2a37a-3068-4aec-94d3-8f587c5e32ad)

#### 2025-01-02-Thu
- `car_sales.drop("<Col Name>", axis=1)` <-- axis=1 represent column
- `pd.Series([1,2,3])` add this as a column in DataFrame rest will fill up as NaN, if python list `[1,2,3]` then error

#### 2025-01-01-Wed
- [x] intro to panda in Jupyter notebook: used `pd.Sereis`, `pd.DataFrame`, `pd.read_csv`, `{data_frame}.to_csv`
- [x] decribe info with pandas using
  - `dtypes`
  - `columns`
  - `.means()`
  - `.info()`
  - `.sum()`
  - ...etc
- [x] select and view data with pandas: 
```python
car_sales.head() # show first 5 rows
car_sales[car_sales["Make"] == "Toyota"] # boolean indexing

car_sales.groupby(["Make"]).mean(numeric_only=True)
car_sales["Odometer (KM)"].hist()
car_sales["Price"] = car_sales["Price"].str.replace('[\$\,\,\.]', '', regex=True)
car_sales["Price"] = car_sales["Price"].astype(int)
car_sales["Price"].plot()
```
- [x] manipulate data with pandas
  - car_sales_drop_missing = car_sales_missing.dropna()
  - car_sales_missing["Odometer"] = car_sales_missing["Odometer"].fillna(car_sales_missing["Odometer"].mean())


#### 2024-12-31-
- [x] activate conda env
- [x] install jupyter pandas numpy matplotlib scikit-learn (issue with higher python version while installing jupyter so had too use `conda create --name jupyter-env python=3.10`)

#### 2024-12-30
- [x] taking machine learning course install mini conda

#### 2022-11-07-Mon

###### Python

Learned from

```
- freeCodeCamp Blog: https://www.freecodecamp.org/news/python-version-on-mac-update/
- freeCodeCamp Python API Development - https://www.youtube.com/watch?v=0sOvCWFmrtA
```

Steps

- Install brew

pyenv to change python version

```sh
brew install pyenv
```

```sh
pyenv install 3.11.0
```

### Set up PATH for pyenv in ZSH

```sh
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
```

```sh
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init --path)"\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
```

### Set up version of python to Global Default

```sh
pyenv global 3.9.2
pyenv versions
```

Install VSCode extension python

### Python virtual env (so that each project use consistent python version)

Go to project and

```sh
# python3 -m venv <name>
python3 -m venv venv
```

Enter interpreter `./venv/bin/python/`

In command line

```sh
source venv/bin/activate
```

will make the command line prefix to be (venv)

```sh
pip install fastapi[all]
```

```sh
pip freeze
```

will show what libs has been installed

```sh
uvicorn main:app --reload
```
