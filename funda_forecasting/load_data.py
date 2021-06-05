'''Contains classes to load data'''

import os
import pandas as pd
class DataLoader(object):

    def __init__(self, base_folder):
        self.base_folder = base_folder

    def load_data(self, file_path):
        return pd.read_csv(os.path.join(self.base_folder,file_path), sep = ';')

    def load_data2(self, file_path):
        return pd.read_csv(os.path.join(self.base_folder,file_path))


