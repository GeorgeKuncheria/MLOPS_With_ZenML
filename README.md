
# Creating a virtual environment and activating it

```bash

conda create -n venv python=3.10 -y

```

```bash

conda activate venv

```


# Important Installations  

```bash

pip install 'zenml[server]'

```

# Other installations

```bash

pip install -r requitements.txt

```


# Running ZenML server for first setup

```bash

zenml init

```



The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

# To run pipeline


```bash

python run_pipeline.py 


```


**When running a pipeline make sure to copy the Trakcer URI(from the CLI) and follow the code below**

```bash

mlflow ui --backend-store-uri "PASTE TRACKER URI"

```

# To run CD 

```bash

python run_deployment.py --config deploy
```



# To disconnect from zenml

```bash

zenml disconnect

```