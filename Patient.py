from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np

import json
from PIL import Image
import matplotlib.pyplot as plt

class Patient(object):

    def __init__(self, data_dict=None):
        self._index = None
        self._race = None
        self._sex = None
        self._age = None
        self._tumor_grade = None
        self._side = None
        self._patho_type = None
        self._m_stage = None
        self._n_stage = None
        self._t_stage = None
        self._surgery = None
        self._tumor_size = None
        self._survival_time = None
        self._oss = None
        self._married_status = None
        self._cancer_sss = None
        self._one = None

        if data_dict:
            if 'index' in data_dict:
                self._index = data_dict['index']
            if 'race' in data_dict:
                self._race = data_dict['race']
            if 'sex' in data_dict:
                self._sex = data_dict['sex']
            if 'age' in data_dict:
                self._age = data_dict['age']
            if 'tumor_grade' in data_dict:
                self._tumor_grade = data_dict['tumor_grade']
            if 'side' in data_dict:
                self._side = data_dict['side']
            if 'patho_type' in data_dict:
                self._patho_type = data_dict['patho_type']
            if 'm_stage' in data_dict:
                self._m_stage = data_dict['m_stage']
            if 'n_stage' in data_dict:
                self._n_stage = data_dict['n_stage']
            if 't_stage' in data_dict:
                self._t_stage = data_dict['t_stage']
            if 'surgery' in data_dict:
                self._surgery = data_dict['surgery']
            if 'tumor_size' in data_dict:
                self._tumor_size = data_dict['tumor_size']
            if 'survival_time' in data_dict:
                self._survival_time =  data_dict['survival_time']
            if 'oss' in data_dict:
                self._oss = data_dict['oss']
            if 'married_status' in data_dict:
                self._married_status = data_dict['married_status']
            if 'cancer_sss' in data_dict:
                self._cancer_sss = data_dict['cancer_sss']
            if 'one' in data_dict:
                self._one = data_dict['one']

    @property   
    def index(self):
        return self._index
    @index.setter
    def index(self, value):
        self._index = value

    @property
    def race(self):
        return self._race
    @race.setter
    def race(self, value):
        self._race = value

    @property
    def sex(self):
        return self._sex
    @sex.setter
    def sex(self, value):
        self._sex = value
    
    @property
    def age(self):
        return self._age
    @age.setter
    def age(self, value):
        # value = float(value)
        self._age = value

    @property
    def tumor_grade(self):
        return self._tumor_grade
    @tumor_grade.setter
    def tumor_grade(self, value):
        self._tumor_grade = value
        
    @property
    def side(self):
        return self._side
    @side.setter
    def side(self, value):
        self._side = value

    @property
    def patho_type(self):
        return self._patho_type
    @patho_type.setter
    def patho_type(self, value):
        self._patho_type = value

    @property
    def m_stage(self):
        return self._m_stage
    @m_stage.setter
    def m_stage(self, value):
        self._m_stage = value
    
    @property
    def n_stage(self):
        return self._n_stage
    @n_stage.setter
    def n_stage(self, value):
        self._n_stage = value

    @property
    def t_stage(self):
        return self._t_stage
    @t_stage.setter
    def t_stage(self, value):
        self._t_stage = value
    
    @property
    def surgery(self):
        return self._surgery
    @surgery.setter
    def surgery(self, value):
        self._surgery = value

    @property
    def tumor_size(self):
        return self._tumor_size
    @tumor_size.setter
    def tumor_size(self, value):
        self._tumor_size = value

    @property
    def survival_time(self):
        return self._survival_time
    @survival_time.setter
    def survival_time(self, value):
        # value = float(value)
        self._survival_time = value
    
    @property
    def oss(self):
        return self._oss
    @oss.setter
    def oss(self, value):
        self._oss = value

    @property
    def married_status(self):
        return self._married_status
    @married_status.setter
    def married_status(self, value):
        self._married_status = value
    
    @property
    def cancer_sss(self):
        return self._cancer_sss
    @cancer_sss.setter
    def cancer_sss(self, value):
        self._cancer_sss = value

    @property
    def one(self):
        return self._one
    @one.setter
    def one(self, value):
        self._one = value

    
    def __repr__(self):
        dict_info = dict()
        dict_info['index'] = self.index
        dict_info['race'] = self._race
        dict_info['sex'] = self._sex
        dict_info['age'] = int(self._age)
        dict_info['tumor_grade'] = self._tumor_grade
        dict_info['side'] = self._side
        dict_info['patho_type'] = self._patho_type
        dict_info['m_stage'] = self._m_stage
        dict_info['n_stage'] = self._n_stage
        dict_info['t_stage'] = self._t_stage
        dict_info['surgery'] = self._surgery
        dict_info['tumor_size'] = self._tumor_size
        dict_info['survival_time'] = int(self._survival_time)
        dict_info['oss'] = self._oss
        dict_info['married_status'] = self._married_status
        dict_info['cancer_sss'] = self._cancer_sss
        dict_info['one'] = self._one

        # patient = json.loads(line, object_hook=patient.Patient)

        return json.dumps(dict_info, ensure_ascii=False)

