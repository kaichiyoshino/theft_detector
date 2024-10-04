#%%
import dataclasses
from datetime import date 

@dataclasses.dataclass
class E:
    transaction_ID: int 
    timestamp: date 
    prosumer: int 
    consumer: int 
    energy: float 
    

