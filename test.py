import asyncio
from loguru import logger
from src.parsing import PhoneCrawler
from src.ml import train_models, predict_price, DataSetBD
from src.schemas import PhoneSchema
from src.config import ML_PHONE_DATASET_PATH
from src.handlers import PhonerHandler


    
if __name__ == '__main__':
    logger.add('logs/test_logs.log', mode='w')
    
    url = "https://www.bakuelectronics.az/catalog/telefonlar-qadcetler/smartfonlar-mobil-telefonlar/xiaomi-poco-m5-6gb128gb-black.html"
    print(PhoneCrawler.start_url(url))
    
    # train_models(data_path=ML_PHONE_DATASET_PATH)
    
    # phone = PhoneSchema(
    #     RAM_GB=12,
    #     brand='Samsung',
    #     camera_mp=200,
    #     CPU_manufacturer='Qualcomm',
    #     NFC=True,
    #     OS='Android',
    #     ROM_GB=256
    # )
    # predict_price(phone.model_dump())
    
    # ds = DataSetBD()
    # print(ds.get_avg_price_seriya("Samsung Galaxy A7 "))
    
    # handler = PhonerHandler()
    # print(handler.ml.plot_stats_for_model("POCO M5 6GB/128GB BLACK"))
    # input()