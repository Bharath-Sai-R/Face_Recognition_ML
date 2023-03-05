import xlwt
from datetime import datetime;
from xlrd import open_workbook;
from xlwt import Workbook
from xlutils.copy import copy
from pathlib import Path

def output(filename, sheet,num, name, present):
    my_file = Path('D:\miniproject\database'+filename+str(datetime.now().date())+'.xls')
    if my_file.is_file():
        rb = open_workbook('D:\miniproject\database'+filename+str(datetime.now().date())+'.xls')
        book = copy(rb)
        sh = book.get_sheet(0)
        # file exists
    else:
        book = xlwt.Workbook()
        sh = book.add_sheet(sheet)
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on',
                         num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    #variables = [x, y, z]
    #x_desc = 'Display'
    #y_desc = 'Dominance'
    #z_desc = 'Test'
    #desc = [x_desc, y_desc, z_desc]
    sh.write(0,0,datetime.now().date(),style1)


    col1_name = 'Name'
    col2_name = 'Present'


    sh.write(1,0,col1_name,style0)
    sh.write(1, 1, col2_name,style0)

    sh.write(num+1,0,name)
    sh.write(num+1, 1, present)
    #You may need to group the variables together
    #for n, (v_desc, v) in enumerate(zip(desc, variables)):
    fullname=filename+str(datetime.now().date())+'.xls'
    book.save('D:\miniproject\database'+fullname)
    return fullname
