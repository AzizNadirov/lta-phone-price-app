import os
from pathlib import Path
import duckdb
import pandas as pd
from loguru import logger
from src.config import ML_PHONE_DATASET_PATH


class DataSetBD:
    """ 
    duckdb conn abstraction on input dataset 
    use `.table` accessor for direct access or special methods.
    """
    instance = None

    def __new__(cls):   
        if cls.instance is None:
            cls.instance = super(DataSetBD, cls).__new__(cls)
        return cls.instance
    
    def sync_from_file(self)->bool:
        """ sync dataset conn from source file """
        logger.warning(f"Sync dataset from file: '{self.path.as_posix()}'")
        try:
            self.table = duckdb.query(f"SELECT * FROM '{self.path}'")
            logger.success("Sync from file: done.")
            return True
        except Exception as e:
            logger.error(f"Error while sync: {e}")
            return False
    
    def __init__(self):
        self.path = Path(ML_PHONE_DATASET_PATH).resolve()
        assert os.path.exists(self.path), f"file {self.path} does not exist"
        self.table = duckdb.query(f"SELECT * FROM '{self.path}'")
        logger.success(f"DataSetDB Initialized.")
        
        
    def get_avg_price_model(self,
                             model: str) -> float | None:
        """ returns average price for given phone 'model' """
        logger.debug(f"Try to get model: {model}")
        tb = 'ds'
        query = f"""
        select AVG(Prices_avg)
        from {tb}
        where model='{model}'
        """
        model_price = self.table.query(virtual_table_name=tb,
                                        sql_query=query).to_df()['avg(Prices_avg)'].dropna()
        
        return model_price.values[0] if not model_price.empty else None
    
    
    def get_price_list_model(self, model: str)->list[float] | None:
        """ returns list of prices for the given model """
        tb = 'ds'
        query = f"""
            select Prices_avg
            from {tb}
            where model ilike '%{model.strip()}%'
            """
            
        model_price = self.table.query(virtual_table_name=tb,
                                        sql_query=query).to_df()['Prices_avg'].dropna()
        
        return model_price.to_list() if not model_price.empty else None