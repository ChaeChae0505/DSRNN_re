# folder에 평가하고 싶은 pt 파일 추가하기
# evluation file에서 평가 데이터 그냥 저장 할 수 있도록 csv
# evluate.py 참고

import os
import subprocess


result = []
path = '/home/a307/collision/220915/Validation_Test/data/5s5d_time/checkpoints'
file_list = os.listdir(path)

file_list_print = [file for file in file_list if file.endswith('.pt')]
file_list_print.sort()
# print(len(file_list_print), file_list_print)

for file in file_list_print:
    call_name = 'python test.py'+' '+'--test_model'+' '+file
    print(call_name)
    subprocess.run(call_name, shell=True)
    result.append(file)
print(result)

