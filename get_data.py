# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:49:00 2019

@author: LBITIND
"""
import pandas as pd
class get_data(object):
    def __init__(self,path):
        self.path=path
        
        
        
    def load_data(self):
      data=pd.read_excel(self.path)
      return data
        
        
        
        
        