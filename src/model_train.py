import os 
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml



log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_train')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_train.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formater=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str)->dict:
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error : %s',e)
        raise



def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        # df.fillna('',inplace=True)
        logger.debug('Data loaded and Nans filled from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise 

def train_model(x_train: np.ndarray,y_train: np.ndarray,params: dict) -> RandomForestClassifier:
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Not same")
        logger.debug("intialise")
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        clf.fit(x_train,y_train)
        return clf
    except ValueError as e:
        logger.error("Value - error during model training: %s",e)
        raise

def save_model(model,file_path: str) ->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to : %s',file_path)
    except FileNotFoundError as e:
        logger.error('File path not found : %s',e)
        raise
    except Exception as e:
        logger.error("occured error")
        raise
def main():
    try:
        # params={'n_estimators':25,'random_state':2}
        params=load_params('params.yaml')['model_train']
        train_data=load_data('./data/processed/train_tfidf.csv')
        # train_data=load_data('./data/processed/train_tfidf.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
        clf =train_model(x_train,y_train,params)
        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error('Faield')

if __name__=='__main__':
    main()

