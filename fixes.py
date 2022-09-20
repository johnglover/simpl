# with open('src/simpl/base.cpp', 'r') as file_:
#     data = file_.readlines()

# data.insert(0, '#include <algorithm>\n')

# with open('src/simpl/base.cpp', 'w') as file_:
#     file_.writelines( data )

# import os

# # rename all .C extensions to .c in src\loris
# for root, dirs, files in os.walk('src/loris/'):
#     for file in files:
#         if file.endswith('.C'):
#             os.rename(os.path.join(root, file_), os.path.join(root, file_.replace('.C', '.c')))