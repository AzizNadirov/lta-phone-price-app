"""object schemas for processes """

from typing import Literal
from pydantic import BaseModel, computed_field



class PhoneSchema(BaseModel):
    """ 
    A Phone schema:
    
        RAM_GB: Phone RAM storage in GB
        ROM_GB: Phone ROM storage in GB
        NFC: Phone Has NFC
        camera_mp: Phone main camera megapixels
        CPU_manufacturer: Phone CPU
        OS: Phone Operation System
        model: Phone's model name
        price: Phone price
        other_specifications: other specifications, normalize them!
    """
    RAM_GB: str|int=None
    ROM_GB: str|int=None
    NFC: bool=None
    camera_mp: str|int=None
    CPU_manufacturer: str=None
    brand: str=None
    OS: str=None
    model: str=None
    price: float=None
    other_specifications: dict=None
    phone_image_link: str=None
    
    
    @computed_field 
    @property
    def RAM_ROM_ratio(self)-> str|float:
        if self.RAM_GB and self.ROM_GB:
            return int(self.RAM_GB) / int(self.ROM_GB)