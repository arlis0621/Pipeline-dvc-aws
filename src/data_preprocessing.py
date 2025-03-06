import os
import logging
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string 
nltk.download('stopwords')
nltk.download('punkt_tab')



log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formater=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)

    text=[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text ]
    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    try:
        logger.debug('Starting preprocessing for dataframe')
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('Column not found : %s',e)
        raise
    except Exception as e:
        logger.error('Error during normalization : %s',e)
        raise
def main(text_column='text',target_column='target'):
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')
        train_processed_data=preprocess_df(train_data,text_column,target_column)
        test_processed_data=preprocess_df(test_data,text_column,target_column)

        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
        logger.debug("Processed data saved")

    except FileNotFoundError as e:
        logger.error('File not found %s',e)
    except Exception as e:
        logger.error('Failed to complete the data process : %s',e)
if __name__=='__main__':
    main()





