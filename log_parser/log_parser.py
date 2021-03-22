#Parsing normalTraffic.txt & anomalousTraffic.txt using Python version 3.6.1
#https://github.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010/blob/master/main.py 

import urllib
import urllib.parse
import io
    
def request_list_to_str(lines: list):
    available_methods = ['GET', 'POST', 'PUT']
    res = []
    for i in range(len(lines)):
        line = lines[i].strip().split(' ')
        if line[0] not in available_methods:
            continue

        url = line[0] + line[1]
        
        if line[0] == "POST" or line[0] == "PUT":
            url += '?' + get_body_of_request(lines, i);
    
        res.append(url.lower())

    return res


def get_body_of_request(lines: list, start_line: int):
    j = 1
    while True:
        if lines[start_line + j].startswith("Content-Length"):
            break
        j += 1
    j += 1
    data = lines[start_line + j + 1].strip()
    return data


def parse(request: list = None, file_in: str = None, file_out: str = None):
    if request:
        lines = request
    elif file_in:
        fin = open(file_in)
        lines = fin.readlines()
        fin.close()
    else:
        return

    res = request_list_to_str(lines)
    print ("finished parse ",len(res)," requests")

    if not file_out:
        return res
    else:
        fout = io.open(file_out, "w", encoding="utf-8")
        for line in res:
            line = urllib.parse.unquote(line).replace('\n','')
            fout.writelines(line + '\n')
        fout.close()


if __name__ == '__main__':
    normal_file = '../data/normalTraffic.txt'
    anomalous_file = '../data/anomalousTraffic.txt'

    normal_parsed = '../data/normal_parsed.txt'
    anomalous_parsed = '../data/anomalous_parsed.txt'

    parse(file_in=normal_file, file_out=normal_parsed)
    parse(file_in=anomalous_file, file_out=anomalous_parsed)
