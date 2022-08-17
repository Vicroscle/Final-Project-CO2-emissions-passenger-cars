### CO2 EMISSIONS PREDICTION OF PASSENGER CARS ###
Prediction based on data from:
European Environment Agency. Monitoring of CO2 emissions from passenger cars – Regulation (EU) 2019/631. url: https://www.eea.europa.eu/data-and-maps/data/co2-cars- emission-20

Workflow:
- Data_Science_CO2_preprocessed_data.py
- Data_Science_model_training.py
- Data_Science_model_predictions.py

The predictions will be made based on the features vehicle pool and make, weigh, wheelbase, axle width, engine capacity and power, fuel type and fuel modus. This kind of prediction could be helpful, as an example, to gain a virtual information of the emissions while developing a new vehicle model before it could be really tested.

The target value that has been chosen is the Enedc (g/km) which is the old emissions measurement protocol, as the new one, the Ewltp (g/km) contains too less values in this dataset. The WLTP was developed to be more representative of real-world driving conditions, so it might be interesting for the future to have a data set which fully contains these values. In this case, target to be change in the code to Ewltp (g/km).

Only the German registrated cars have been chosen due to the high amount of instances. For further learning runs in the future, all the countries could be considered. In this case, the Feature "MS" Member State shouldn´t be removed in the pre-processing .py.

The vehicle make has been target encoded, so a dictionary has to be created to make new predictions.

The graphic/ploting analyse ,including some hyperparameter research and evaluation,can be found at the Jupyter Notebooks.

Currently the KNN Model with the hyperparameters is the one minimazing at best the Mean Squared error.

### REPOSITORY DEVELOPEMENT SETUP ###

Open the [integrated terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) and run the setup script for your OS (see below). This will install a [Python virtual environment](https://docs.python.org/3/library/venv.html) with all packages specified in `requirements.txt`.

### Linux and Mac Users
1. run the setup script: `./setup.sh` or `sh setup.sh`
2. activate the python environment: `source .venv/bin/activate`
### Windows Users
1. run the setup script `.\setup.ps1`
2. activate the python environment: `.\.venv\Scripts\Activate.ps1`

3. run example code: `python src/Example.py`

Troubleshooting:

- If your system does not allow to run powershell scripts, try to set the execution policy: `Set-ExecutionPolicy RemoteSigned`, see https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/
- If you still cannot run the setup.ps1 script, open it and copy all the commands step by step in your terminal and execute each step
