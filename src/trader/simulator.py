#%%
from blockchain import Chain, Block, Caliculator
from energy_trading import E
import numpy as np 
import time
from datetime import date
from pathlib import Path 
import hashlib 
import json 


class Simulator:
    def __init__(self, simulator):
        self.simulator_version = "1.0.0"
        self.simulator = simulator 
        self.simulator_dict = {
            "block_sim": self.blockchain_simulation,
        }
        self.house_num = 10
    
    def blockchain_simulation(self, transaction_limit: int):
        _chain = Chain()
        transaction_count = 0
        while transaction_count < transaction_limit:
            transaction = self.energy_trading_generator(hashlib.sha256(str(transaction_count).encode("utf-8")), np.random.randint(1, 10), np.random.randint(1, 10), np.random.uniform(0, 1))
            block = Block(1, 1, 1, 1, 1, transaction=transaction)
            time.sleep(10)
            _chain.add_block(block, True)
            print(_chain.chain[transaction_count].transaction)
            transaction_count += 1
    
    def energy_trading_generator(self, trasaction_ID: int, prosumer: int, consumer: int, energy: float):
        trading = E(trasaction_ID, time.time(), prosumer, consumer, energy)
        return trading 
        

simulator = Simulator("blockchain")
simulator.blockchain_simulation(10)

#%%