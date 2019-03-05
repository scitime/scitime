The preferred way to contribute to scitime is to fork the main repository on GitHub, then submit a “pull request” (PR) - as done for scikit-learn contributions:

- Create an account on GitHub if you do not already have one.
- Fork the project repository: click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. 
- Clone your fork of the scitime repo from your GitHub account to your local disk:
```$ git clone git@github.com:YourLogin/scitime.git
$ cd scitime
# Install library in editable mode:
$ pip install --editable .
```


- Create a branch to hold your development changes:

```
$ git checkout -b my-feature
```


and start making changes. Always use a feature branch. It’s good practice to never work on the masterbranch!

- Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using git add and then git commit files:
```
$ git add modified_files
$ git commit
```

- to record your changes in Git, then push the changes to your GitHub account with:
```
$ git push -u origin my-feature
```


- Follow GitHub instructions to create a pull request from your fork. 

Some quick additional notes:
- We use appveyor and travis.ci for our tests
- We try to follow the PEP8 guidelines (using flake8, ignoring codes E501 and F401)
