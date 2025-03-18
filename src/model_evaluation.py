import os 
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,roc_auc_score
import logging
from dvclive import Live
import yaml


log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evaluation.log')
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






def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model=pickle.load(file)
        logger.debug("model loaded")
        return model
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except Exception as e:
        logger.error("unexpected error occured while loading the model")
        raise

def load_data(file_path :str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug('Data loaded from')
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file")
        raise
    except Exception as e:
        logger.error('unexpected error occured')
        raise


def evaluate_model(clf,x_test :np.ndarray,y_test:np.ndarray)->dict:
    y_pred=clf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    metrics_dict={'accuracy':accuracy}
    logger.debug("model evaluation metrics calculated")
    return metrics_dict


def save_metrics(metrics :dict,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug('Metrics saved to %s',file_path)
    except Exception as e:
        logger.error('Error occured while saving %s',e)
        raise

def main():
    try :
        params=load_params('params.yaml')['model_train']

        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/processed/test_tfidf.csv')

        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values

        metrics=evaluate_model(clf,x_test,y_test)

        with Live(save_dvc_exp=False) as dtklive:
            dtklive.log_metric('accuracy',accuracy_score(y_test,y_test))
            dtklive.log_params(params)

        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error("failed")
        raise

if __name__=='__main__':
    main()

