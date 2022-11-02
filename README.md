# Full review of major time series analysis tasks

# Setup python environment

* create environment
`python3 -m venv env`  

* activate environment

`source env/bin/activate` (*nix)  

`env\Scripts\activate.bat` (windows)  

* install all dependencies

`pip install -r requirements.txt`

# Contents

> All materials also have solution versions, you can identify those by _solution.ipynb
prefixes

**ts_walkthrough** - main jupyter notebook with all theoretical materials and tasks, you
should start here

**arima.ipynb**  - separate jupyter notebook for arima task

**supervised.ipynb** - separate jupyter notebook for supervised ML task

**plotting.py**  - helper script for plotting utilities (used in notebooks as is, not a
task)

**dataset.py** - helping script for data handling utilities (used in notebooks as is, not a
task)

**data** - contains all series, used in tutorial

**test.md** - short test on basic time series concepts
