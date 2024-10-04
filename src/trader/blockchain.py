#%%
import hashlib
import json 
from uuid import uuid4
from time import time
from energy_trading import E


class Block:
    def __init__(self, previous_hash, merkle_root, timestamp, difficulty, nonce, transaction: E):
        self.__block_header = {"previous_hash": previous_hash, "merkle_root": merkle_root, "timestamp": timestamp, "difficulty": difficulty, "nonce": nonce}
        self.__block_transaction = transaction
        self.__hash = 0
    
    @property
    def header(self):
        return self.__block_header

    @header.setter
    def header(self):
       pass 

    @property 
    def transaction(self):
        return self.__block_transaction
    
    @transaction.setter
    def transaction(self):
        pass
    

class Chain:
    def __init__(self):
        self.version = "1.0.0"
        self.consensus_algorithm = "PoW"
        self.__chain = []
        self.__current_transactions = []

    def __str__(self):
        return f"Chain version is {self.version}"
    
    def __repr__(self):
        return f"Chain: {self.version}\n Consensus_Algorithm: {self.consensus_algorithm}"
    
    @property
    def chain(self):
        return self.__chain
    
    @chain.setter
    def chain(self, block_list:list):
        if not isinstance(block_list, list):
            raise ValueError("Block list must be a list")
        self.__chain = block_list
            
    
    def add_block(self, block:Block, agreement:bool):
        """
        Add a block to the chain
        """
        if agreement:
            self.__chain.append(block)


class Caliculator:
    def __init__(self):
        pass

    def p_o_w(self):
        pass

    def p_o_s(self):
        pass

class User:
    def __init__(self):
        pass
