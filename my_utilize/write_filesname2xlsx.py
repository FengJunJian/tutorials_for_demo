import os
import xlwt  # 操作excel模块
import sys

root='E:\\paper\\半监督船舶检测en'
search_filedir='E:\\fjj\\keras-yolo3-master\\save_ships'
file_path = os.path.join(root,'Fig.semi图片序列.xls')  # sys.path[0]为要获取当前路径，filenamelist为要写入的文件
f = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 新建一个excel
sheet = f.add_sheet('Sheet1')  # 新建一个sheet
pathDir = os.listdir(search_filedir)  # 文件放置在当前文件夹中，用来获取当前文件夹内所有文件目录

i = 0  # 将文件列表写入test.xls
for s in pathDir:
    sheet.write(i, 0, s)  # 参数i,0,s分别代表行，列，写入值
    i = i + 1

print(file_path)
print(i)  # 显示文件名数量
f.save(file_path)