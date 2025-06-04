import asyncio, os, json, time, traceback, re
from functools import reduce
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv
from loguru import logger
from src.schemas import PhoneSchema
from src.config import PARSING_LLM_ATTEMPTS, PARSING_LLM_WAIT




def merge_dicts(dict_list: list[dict])->dict:
    """ merge list of dicts """
    def merge_two(d1: dict, d2: dict):
        return {k: v if v is not None else d1.get(k) for k, v in {**d1, **d2}.items()}
    return reduce(merge_two, dict_list, {})


def extract_json(data_string: str) -> dict | list | None:
    """Try to extract JSON from a string using multiple patterns sequentially; return None if extraction fails."""
    patterns = [
        r'```json\n(.*?)\n```',
        r'(\{.*\}|\[.*\])'
    ]
    for pattern in patterns:
        try:
            match = re.search(pattern, data_string, re.DOTALL)
            if match:
                json_data = match.group(1).strip()
                return json.loads(json_data)
        except Exception as e:
            logger.warning(f"Error while extracting JSON using pattern {pattern}: {e}")
            logger.warning(f"Data: {data_string}")
    return None






class PhoneCrawler:
    """ LLM based product specification crawler. """

    def __init__(self):
        load_dotenv(override=True)
    
    @classmethod
    async def _extract(cls, url: str)-> dict | None:
        for attempt in range(PARSING_LLM_ATTEMPTS):
            try:
                async with AsyncWebCrawler() as crawler:
                    strategy = cls._build_strategy()
                    result = await crawler.arun(
                        url=url,
                        bypass_cache=True,
                        extraction_strategy=strategy,
                    )
                    logger.debug(f"Crawler status code: {result.status_code}")
                    result_json = extract_json(result.extracted_content)
                    logger.debug(f"Crawler parsed result all: {result_json}")
                    if isinstance(result_json, dict) and result_json.get('error') or not result_json:
                        logger.error(f"Unable to parse page with url: {url}; attempt: {attempt+1}/{PARSING_LLM_WAIT}; error: {result_json}")
                        logger.debug(f"Wait for {PARSING_LLM_WAIT} secunds.")
                        time.sleep(PARSING_LLM_WAIT)
                        continue
                    if isinstance(result_json, list):
                        result_json = [d for d in result_json if not d.get('error')]
                        logger.debug(f"Crawler parsed result: {result_json}")
                        if not result_json:
                            raise Exception("Unable to parse page with url: " + url)
                        
                        if isinstance(result_json, list):
                            return merge_dicts(result_json)
                        elif isinstance(result_json, dict):
                            return result_json
                        else:
                            raise ValueError(f"Unexpected result: '{result_json}' of type '{type(result_json)}' ")

            except Exception as e:
                logger.error(f"Unable to parse page with url: {url}; attempt: {attempt+1}/{PARSING_LLM_WAIT}; error: {e}")
                logger.error(traceback.format_exc())
                logger.debug(f"Wait for {PARSING_LLM_WAIT} secunds.")
                time.sleep(PARSING_LLM_WAIT)
                continue
            
        return None
    
    @classmethod
    def start_url(cls, url: str):
        """ 
        start crawler for given url 
        
        Arguments:
            - url: str - url to parse
        Returns:
            - dict: parsed phone for prediction
        """
        logger.debug(f"Crawler started with: url: {url}")
        return asyncio.run(cls._extract(url))
    
    @classmethod
    def _build_strategy(cls):
        prompt = """
            You are a parser bot. Yor main goal is to extact product characteristics from product page. 
            Try to translate specification names and values to english standard format. 
            Try to normalize values: if OS = 'Android 12 MIUI X' then value will be main part: 'Android'
            For example, if specification name is "Оперативная память: 16 ГБ", translate it to "ram: 16 GB". If output response is very huge,
            then keep only fields that have real values. Response according to the schema!
            Schema description:
            
            RAM_GB: Phone RAM storage in GB
            ROM_GB: Phone ROM storage in GB
            NFC: Phone Has NFC
            camera_mp: Phone main camera megapixels
            CPU_manufacturer: Phone CPU
            OS: Phone Operation System
            model: Phone's model name. Model name does not contains specifications. For example, Phone name is "POCO M5 6GB/128GB BLACK", but its Model name is "POCO M5"
            price: Phone price
            other_specifications: other specifications, normalize them!
            phone_image_link: direct image link if exists on the page
            """
        
        extraction_strategy_oai = LLMExtractionStrategy(
                provider = f"azure/{os.environ['GPT41_API_DEPLOYMENT_NAME']}", 
                api_base=os.environ['GPT41_API_BASE'],
                api_token=os.environ['GPT41_API_KEY'],
                api_version=os.environ['GPT41_API_VERSION'],
                schema=PhoneSchema.model_json_schema(),
                extraction_type="schema",
                instruction=prompt,
                overlap_rate=1,
        )
        
        return extraction_strategy_oai
    
    