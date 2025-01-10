# Learn in Public

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
