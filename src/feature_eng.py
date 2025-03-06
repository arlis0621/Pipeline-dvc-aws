import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('feature_eng')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'feature_eng.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formater=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#loading the data from interim
def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data loaded and Nans filled from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s',e)
        raise 

def apply_tfidf(train_data: pd.DataFrame ,test_data:pd.DataFrame,max_features: int)->tuple :
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)
        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values
        y_test=test_data['target'].values

        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.transform(x_test)

        train_df=pd.DataFrame(x_train_bow.toarray())

        train_df['label']=y_train

        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test

        logger.debug('Bag of words applied and transformed')
        return train_df,test_df
    except Exception as e:
        logger.error('Error during bag of words : %s',e)
        raise

def save_data(df: pd.DataFrame,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('data saved')
    except Exception as e:
        logger.error('Unexpected error occured while saving the data : %s',e)
        raise

def main():
    try:
        max_features=50
        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')

        train_df,test_df=apply_tfidf(train_data,test_data,max_features)

        save_data(train_df,os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df,os.path.join("./data","processed","test_tfidf.csv"))

    except Exception as e:
        logger.error("Failed")
if __name__=='__main__':
    main()
    