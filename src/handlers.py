""" service handlers """
from typing import TYPE_CHECKING
from loguru import logger
from src.ml import DataSetBD, PhonePricePredictor, predict_price, plot_price_distributions
from src.parsing import PhoneCrawler
from src.schemas import PhoneSchema
from src.config import ML_PHONE_DATASET_PATH
if TYPE_CHECKING:
    from matplotlib.figure import Figure



class MLHandler:
    def __init__(self):
        self.dataset = DataSetBD()
        self._predictor = PhonePricePredictor()
        self._setup()
        # todo: can retrain based on the cron or signal (for web)
        
    def _setup(self):
        loaded = self._predictor.load_models()
        if not loaded:
            logger.debug("Models not loaded. Try to train.")
            self.train()
        logger.success("Models loaded!")
        
    def plot_stats_for_model(self, model: str)->'Figure':
        prices = self.dataset.get_price_list_model(model=model)
        if not prices:
            logger.error(f"Not found records for '{model}' model.")
            return None
        
        return plot_price_distributions(price_data={model: prices})
        
    
    def train(self):
        logger.info("Start training ML models.")
        models = self._predictor.train_and_test(file_path=ML_PHONE_DATASET_PATH)
        if models:
            logger.success("Models training finished.")
        else:
            logger.error("Models training failed.")
            
    def predict_for(self, 
                    phone: dict | PhoneSchema) -> float | None:
        """ predict price for given phone """
        if not isinstance(phone, (PhoneSchema, dict)):
            logger.error(f"Phone for predicting must be instance of dict or PhoneSchema, but got: '{type(phone)}'")
            return None
        
        logger.info(f"Predicting price for {phone}")
        if isinstance(phone, PhoneSchema):
            logger.debug("Dumping model instance...")
            phone = phone.model_dump()
            
        logger.debug("Predicting...")
        predicted_price = self._predictor.predict_price(phone_specs=phone)
        logger.debug(f"Got prediction: {predicted_price}")
        return predicted_price



class PhonerHandler:
    """  """
    def __init__(self):
        """  """
        self.ml = MLHandler()
        self.crawler = PhoneCrawler()
        
    def predict_from_url(self, 
                         url: str,
                         attempts: int=3)->float | None:
        for attempt in range(attempts):
            logger.debug(f"Attempt: {attempt}/{attempts}")
            parsed_data = self.crawler.start_url(url=url)
            if not parsed_data:
                logger.error(f"Unable to parse anything from '{url}'.")
                continue
            
            try:
                parsed_data = PhoneSchema(**parsed_data)
            except Exception as e:  # replace with pydantic's excp
                logger.error(f"Invalid data for Phone: '{parsed_data}'.\n Schema: {PhoneSchema}")
                logger.error(e)
                continue
            
            logger.debug("Predicting...")
            predicted_price = self.ml.predict_for(parsed_data)
            if not predicted_price:
                logger.error(f"Unable to predict price for '{url}' with data '{parsed_data}'")
                continue
            
            return predicted_price
        
        logger.error(f"Failed after {attempts} attempts.")
        return None
            