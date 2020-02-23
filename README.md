# disaster_response_pipeline
Udacity Data Enigneering Project
# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Motivation : 
The current analysis was performed, to classify disaster messages data.

### File Description : 
.
+-- app     
¦   +-- run.py                           # Flask file that runs app
¦   +-- templates   
¦       +-- go.html                      # Classification result page of web app
¦       +-- master.html                  # Main page of web app    
+-- data                   
¦   +-- disaster_categories.csv          # categories  data
¦   +-- disaster_messages.csv            # messages data
¦   +-- process_data.py                  # script to clean data 
+-- models
¦   +-- train_classifier.py              # script to train model and classify           
+-- README.md
