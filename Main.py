import os
from ML_01 import ProcessUnit
import numpy as np
# Specify the folder path
folder_path = "C:\\Users\\muham\\OneDrive\\Desktop\\project\\Ai_project-main\\Data"

# List all files in the folder3
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Print the list of .csv files
print(csv_files)
user = None
try:
    user = int(input("show Output Graph press: 0\nwithout showing Graph press: 1 \nEnter your choice here: "))
except:
    print("Invalied Input")
    exit()

if user!=0 and user!=1:
    print("wrong Input")
    exit()
else:
    pass

for item in csv_files:
    print("FileName: ", user)
    try:
        if user != None:
            ProcessUnit(item,folder_path,user)
    except:
        pass