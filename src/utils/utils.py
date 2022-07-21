import zipfile
import os
import argparse
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))

'''
Download compressed raw data package into
'utils/raw_data' Folder and uncompressed
'''
def download_package():
    pass

def unzip_folder(zip_folder, destination = None):
        """
        Args:
            zip_folder (string): zip folder to be unzipped
            destination (string): path of destination folder
            pwd(string): zip folder password
        """
        if destination == None:
            file_name = os.path.splitext(os.path.basename(zipfile_folder))[0]
            dir_name = os.path.dirname(os.path.dirname(os.path.realpath(zipfile_folder)))
            destination = dir_name + '/utils/data/' + file_name

        with zipfile.ZipFile(zip_folder) as zf:
            zf.extractall(
                destination, pwd = None)

'''
Make sure destination folder contains raw data
'''       
def generate_classification_label(zip_folder,split_test):
    file_name = os.path.splitext(os.path.basename(zipfile_folder))[0]
    dir_name = os.path.dirname(os.path.dirname(os.path.realpath(zipfile_folder)))
    destination = dir_name + '/utils/data/' + file_name
    filename = destination + "/FT.txt"
    data_points = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
            point = [float(i) for i in lines.split(',')]
            point[0] = str(int(point[0]))+'.jpg'
            data_points.append(point)  
    name = ['file_name','x','y','z','fx-ext','fy-ext','fz-ext','tx-ext','ty-ext','tz-ext','fx-built-in','fy-built-in','fz-built-in','tx-built-in','ty-built-in','tz-built-in']
    data_info = pd.DataFrame(columns = name, data = data_points)

    x = np.asarray(data_info.loc[:, 'x'])
    x = x.astype(int)
    y = np.asarray(data_info.loc[:, 'y'])
    y = y.astype(int)

    ## generate several resolution labels
    # 10mm space
    class_code = x//10 + 2 * (y//10)
    data_info["10mm-labels"] = class_code

    # 5mm space
    class_code = x//5 + 4 * (y//5)
    data_info["5mm-labels"] = class_code

    # 2mm space
    class_code = x//2 + 10 * (y//2)
    data_info["2mm-labels"] = class_code
    
    train_info, test_info = train_test_split(data_info,test_size = split_test)


    csv_file_name = destination + '/train_class_label.csv'
    train_info.to_csv(csv_file_name,index = False)
    csv_file_name = destination + '/test_class_label.csv'
    test_info.to_csv(csv_file_name,index = False)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
'''
Normalise x range in (0 , 1)
Normalise y range in (0 , 1)
Normalise z range in (0 , 1)
Normalise Fz range in (0 , 1)
'''
def generate_regression_data(zip_folder,split_test):
    file_name = os.path.splitext(os.path.basename(zipfile_folder))[0]
    dir_name = os.path.dirname(os.path.dirname(os.path.realpath(zipfile_folder)))
    destination = dir_name + '/utils/data/' + file_name
    filename = destination + "/FT.txt"
    data_points = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
            point = [float(i) for i in lines.split(',')]
            point[0] = str(int(point[0]))+'.jpg'
            data_points.append(point)  
    name = ['file_name','x','y','z','fx-ext','fy-ext','fz-ext','tx-ext','ty-ext','tz-ext','fx-built-in','fy-built-in','fz-built-in','tx-built-in','ty-built-in','tz-built-in']
    data_info = pd.DataFrame(columns = name, data = data_points)

    x = np.asarray(data_info.loc[:, 'x'])
    y = np.asarray(data_info.loc[:, 'y'])
    z = np.asarray(data_info.loc[:, 'z'])
    fz = np.asarray(data_info.loc[:, 'fz-built-in'])

    data_info["Normalized_x"] = normalization(x)
    data_info["Normalized_y"] = normalization(y)
    data_info["Normalized_z"] = normalization(z)
    data_info["Normalized_fz"] = normalization(fz)

    train_info, test_info = train_test_split(data_info,test_size = split_test)

    csv_file_name = destination + '/train_data_point.csv'
    train_info.to_csv(csv_file_name,index = False)
    csv_file_name = destination + '/test_data_point.csv'
    test_info.to_csv(csv_file_name,index = False)

def generate_regression_data():
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    filename = dir_path + "/data/soft_finger_01/FT.txt"
    data_points = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
            point = [float(i) for i in lines.split(',')]
            point[0] = str(int(point[0]))+'.jpg'
            data_points.append(point)  
    name = ['file_name','x','y','z','fx-ext','fy-ext','fz-ext','tx-ext','ty-ext','tz-ext','fx-built-in','fy-built-in','fz-built-in','tx-built-in','ty-built-in','tz-built-in']
    data_frame = pd.DataFrame(columns = name, data = data_points)

    csv_file_name = dir_path + "/data/soft_finger_01" + "/data_frame.csv"
    data_frame.to_csv(csv_file_name,index = False)

def parse_args():
    parser = argparse.ArgumentParser(description='Train VisTac CNN')
    # DataSet
    parser.add_argument('--zip-folder-path', type=str, help='Path to zip folder')
    parser.add_argument('--model-type', type=str, default='classification', help='classification or regression')
    parser.add_argument('--split',type=float, default=0.3, help='fraction of data held for test')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    generate_regression_data()
    # print(dir_path)

    # args = parse_args()

    # zipfile_folder = args.zip_folder_path
    # test_size = args.split
    # unzip_folder(zipfile_folder)

    # if args.model_type == 'classification':
    #     generate_classification_label(zipfile_folder, test_size)

    # if args.model_type == 'regression':
    #     generate_regression_data(zipfile_folder, test_size)