
import os
import sys
import json
import csv
import pickle
sys.path.append(sys.path[0])
sys.path.append(sys.path[0]+'/../')

import numpy as np 
import pandas as pd 
from Patient import Patient

def read_original_data():
    data = pd.read_csv('./bladder_cancer.csv', encoding='gbk')
    print('the shape of input data is ', data.shape)
    # print(data.columns.tolist())
    columns = data.columns.tolist()
    #columns = ['Race', 'Sex', 'Age', 'Tumor Grade', 'Side', 'Pathology Type', 'M stage', 'N stage', 'T stage', 'Surgery', 
    #   'Tumor size (mm)', 'Survival time (month)', 'Overall survival status', 'Married status', 
    #   'Cancer specific survival status (暂时不用)', '1']
    # print(data.iloc[0,:])

    txt_path ='./bladder_cancer.txt'
    if os.path.exists(txt_path):
        if isinstance(txt_path, str):
            patients = list()
            print('Start to read the txt file ......')
            with open(txt_path,'r') as f:
                for line in f.readlines():
                    unit = json.loads(line, object_hook=Patient)
                    patients.append(unit)
            print('Finish the reading process of txt file ......')
    else:
        patients = list()
        f = open(txt_path,'w')
        for i in np.arange(data.shape[0]):
            # for j in np.arange(data.shape[1]):
            #     print(type(data.loc[i, columns[j]]))
            unit = Patient()
            unit.index = str(i)
            unit.race = data.loc[i, columns[0]]
            unit.sex = data.loc[i,columns[1]]
            unit.age =  data.loc[i,columns[2]]
            unit.tumor_grade =  data.loc[i,columns[3]]
            unit.side = data.loc[i,columns[4]]
            unit.patho_type = data.loc[i,columns[5]]
            unit.m_stage = data.loc[i,columns[6]]
            unit.n_stage = data.loc[i,columns[7]]
            unit.t_stage = data.loc[i,columns[8]]
            unit.surgery = data.loc[i,columns[9]]
            unit.tumor_size = data.loc[i,columns[10]]
            unit.survival_time = data.loc[i,columns[11]]
            unit.oss = data.loc[i,columns[12]]
            unit.married_status = data.loc[i,columns[13]]
            unit.cancer_sss = data.loc[i,columns[14]]
            unit.one = data.loc[i,columns[15]]

            if i % 1000 == 0:
                print(str(unit))
            patients.append(unit)
            f.write(str(unit)+'\n')
        
        f.close()
    print('the number of patients is ', len(patients))

    return patients

def string_to_category():
    txt_path ='./bc_value.txt'
    if os.path.exists(txt_path):
        if isinstance(txt_path, str):
            patient_values = list()
            print('Start to read bc_value.txt ......')
            with open(txt_path,'r') as f:
                for line in f.readlines():
                    unit = json.loads(line, object_hook=Patient)
                    patient_values.append(unit)
            print(patient_values[0])
            print('Finish the reading process of bc_value.txt ......')
    else:
        target_file = './target.pickle'
        f = open(target_file, 'rb')
        targets = pickle.load(f)
        f.close()
        target_keys = list(targets.keys())
        print(targets)
        print(target_keys)

        patient_values = read_original_data()
        print(patient_values[0])

        for p in patient_values:
            
            assert target_keys[0] == 'Race'
            if p.race in list(['unknown', 'Unknown']):
                p.race = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[0]])):
                    if p.race == targets[target_keys[0]][i][0]:
                        p.race = targets[target_keys[0]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.race with wrong category-{}'.format(p.race))

            assert target_keys[1] == 'Sex'
            if p.sex in list(['unknown', 'Unknown']):
                p.sex = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[1]])):
                    if p.sex == targets[target_keys[1]][i][0]:
                        p.sex = targets[target_keys[1]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.sex with wrong category')

            assert target_keys[2] == 'Tumor Grade'
            if p.tumor_grade in list(['unknown', 'Unknown']):
                p.tumor_grade = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[2]])):
                    if p.tumor_grade == targets[target_keys[2]][i][0]:
                        p.tumor_grade = targets[target_keys[2]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.tumor_grade with wrong category')

            assert target_keys[3] == 'Side'
            if p.side in list(['unknown', 'Unknown']):
                p.side = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[3]])):
                    if p.side == targets[target_keys[3]][i][0]:
                        p.side = targets[target_keys[3]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.side with wrong category')

            assert target_keys[4] == 'Pathology Type'
            if p.patho_type in list(['unknown', 'Unknown']):
                p.patho_type = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[4]])):
                    if p.patho_type == targets[target_keys[4]][i][0]:
                        p.patho_type = targets[target_keys[4]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.patho_type with wrong category')

            assert target_keys[5] == 'M stage'
            if p.m_stage in list(['unknown', 'Unknown']):
                p.m_stage = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[5]])):
                    if p.m_stage == targets[target_keys[5]][i][0]:
                        p.m_stage = targets[target_keys[5]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.m_stage with wrong category')

            assert target_keys[6] == 'N stage'
            if p.n_stage in list(['unknown', 'Unknown']):
                p.n_stage = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[6]])):
                    if p.n_stage == targets[target_keys[6]][i][0]:
                        p.n_stage = targets[target_keys[6]][i][-1]; temp=True 
                if not temp:
                    raise ValueError('p.n_stage with wrong category')

            assert target_keys[7] == 'T stage'
            if p.t_stage in list(['unknown', 'Unknown']):
                p.t_stage = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[7]])):
                    if p.t_stage == targets[target_keys[7]][i][0]:
                        p.t_stage = targets[target_keys[7]][i][-1]; temp=True 
                if not temp:
                    raise ValueError('p.t_stage with wrong category')

            assert target_keys[8] == 'Surgery'
            if p.surgery in list(['unknown', 'Unknown']):
                p.surgery = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[8]])):
                    if p.surgery == targets[target_keys[8]][i][0]:
                        p.surgery = targets[target_keys[8]][i][-1]; temp=True 
                if not temp:
                    raise ValueError('p.surgery with wrong category')

            if p.tumor_size in list(['unknown', 'Unknown']):
                p.tumor_size = np.nan
            elif float(p.tumor_size) >= 0:
                p.tumor_size = float(p.tumor_size)
            else:
                raise ValueError('p.tumor_size has wrong data type-{}'.format(p.tumor_size))

            assert target_keys[9] == 'Married status'
            if p.married_status in list(['unknown', 'Unknown']):
                p.married_status = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[9]])):
                    if p.married_status == targets[target_keys[9]][i][0]:
                        p.married_status = targets[target_keys[9]][i][-1]; temp=True 
                if not temp:
                    raise ValueError('p.married_status with wrong category')
            
            assert target_keys[10] == 'Overall survival status'
            if p.oss in list(['unknown', 'Unknown']):
                p.oss = np.nan
            else:
                temp = False
                for i in np.arange(len(targets[target_keys[10]])):
                    if p.oss == targets[target_keys[10]][i][0]:
                        p.oss = targets[target_keys[10]][i][-1]; temp=True
                if not temp:
                    raise ValueError('p.oss with wrong category')

        print(patient_values[0])
        f = open(txt_path, 'w')
        for p in patient_values:
            f.write(str(p)+'\n')
        f.close()
    return patient_values

def selected_unmiss_data():
    patient_values = string_to_category()
    s_data = list()
    for p in patient_values:
        temp = list()
        temp.append('patient_'+str(p.index))

        if np.isnan(p.race): continue
        temp.append(p.race)

        if np.isnan(p.sex): continue
        temp.append(p.sex)

        if np.isnan(p.age): continue
        temp.append(p.age)

        if np.isnan(p.tumor_grade): continue
        temp.append(p.tumor_grade)

        if np.isnan(p.side): continue
        temp.append(p.side)

        if np.isnan(p.patho_type): continue
        temp.append(p.patho_type)

        if np.isnan(p.m_stage): continue
        temp.append(p.m_stage)

        if np.isnan(p.n_stage): continue
        temp.append(p.n_stage)

        if np.isnan(p.t_stage): continue
        temp.append(p.t_stage)

        if np.isnan(p.surgery): continue
        temp.append(p.surgery)

        if np.isnan(p.tumor_size): continue
        temp.append(p.tumor_size)

        if np.isnan(p.married_status): continue
        temp.append(p.married_status)

        if np.isnan(p.survival_time): continue
        if np.isnan(p.oss): continue

        # if int(p.survival_time) < 60 and  int(p.oss) == 0: continue
        # elif int(p.survival_time) < 60:
        if int(p.survival_time) < 60:
            temp.append(1)
        elif int(p.survival_time) >= 60:
            temp.append(0)
        else:
            raise ValueError('the survival time of patient-{} is {}, which has wrong value format!'.format(p.index, p.survival_time))

        assert len(temp) == 14, 'the patient data has wrong size and lost some feature!'
        s_data.append(temp)
    
    print('the patient with complete data: ', s_data[0])

    f = open('./unmiss_for_models_undel.csv','w')
    csv_writer = csv.writer(f)
    columns = ['Index','Race','Sex','Age','Tumor grade','Side','Pathology type','M stage','N stage','T stage','Surgery','Tumor size','Married status','Label']
    assert len(columns) == 14, 'the columns data has wrong size and lost some feature!'
    csv_writer.writerow(columns)
    for s in s_data:
        csv_writer.writerow(s)
    f.close()

def selected_miss_data():
    patient_values = string_to_category()
    s_data = list()
    for p in patient_values:
        temp = list()
        temp.append('patient_'+str(p.index))

        temp.append(p.race)
        temp.append(p.sex)
        temp.append(p.age)
        temp.append(p.tumor_grade)
        temp.append(p.side)
        temp.append(p.patho_type)
        temp.append(p.m_stage)
        temp.append(p.n_stage)
        temp.append(p.t_stage)
        temp.append(p.surgery)

        if np.isnan(p.tumor_size): continue
        temp.append(p.tumor_size)

        temp.append(p.married_status)

        if np.isnan(p.survival_time): continue
        if np.isnan(p.oss): continue

        if int(p.survival_time) < 60 and  int(p.oss) == 0: continue
        elif int(p.survival_time) < 60:
        # if int(p.survival_time) < 60:
            temp.append(1)
        elif int(p.survival_time) >= 60:
            temp.append(0)
        else:
            raise ValueError('the survival time of patient-{} is {}, which has wrong value format!'.format(p.index, p.survival_time))

        assert len(temp) == 14, 'the patient data has wrong size and lost some feature!'
        s_data.append(temp)
    
    print('the patient with complete data: ', s_data[0])

    f = open('./miss_for_models_del.csv','w')
    csv_writer = csv.writer(f)
    columns = ['Index','Race','Sex','Age','Tumor grade','Side','Pathology type','M stage','N stage','T stage','Surgery','Tumor size','Married status','Label']
    assert len(columns) == 14, 'the columns data has wrong size and lost some feature!'
    csv_writer.writerow(columns)
    for s in s_data:
        csv_writer.writerow(s)
    f.close()

def selected_original_data(f_file, to_path):
    all_original_data = pd.read_csv('./bladder_cancer.csv', encoding='gbk')
    from_data = pd.read_csv(f_file, encoding='utf8')
    from_data_col1 = from_data.iloc[:,0].tolist()
    from_data_list = [int(s.split('_')[-1]) for s in from_data_col1]
    selected_original_data =  all_original_data.iloc[from_data_list]
    selected_original_data.to_csv(to_path,index=None)
    print('Completing this task ......')
    

if __name__ == '__main__':
    # read_original_data()
    # string_to_category()
    # selected_unmiss_data()
    # selected_miss_data()
    f_path = ''
    to_path = ''
    selected_original_data(f_path, to_path)
