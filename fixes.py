with open('src/simpl/base.cpp', 'r') as file_:
    data = file_.readlines()

data.insert(0, '#include <algorithm>\n')

with open('src/simpl/base.cpp', 'w') as file_:
    file_.writelines( data )