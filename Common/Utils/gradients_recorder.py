import csv 

def write_to_csv(path:str, mode='a', data=[]):
    with open(path, mode, newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data)

def write_to_txt(path:str, mode='a', data=[]):
    with open(path, mode) as f:
        f.write(''.join(str(i)+' ' for i in data)+'\n')
