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
