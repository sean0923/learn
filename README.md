# Learn in Public

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
