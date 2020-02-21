import xlrd
import csv

def csv_from_excel():   
    workbook = xlrd.open_workbook('trial.xlsx')
    sh = workbook.sheet_by_name('Sheet1')
    dataset = open('dataset.csv', 'w')
    write = csv.writer(dataset, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        write.writerow(sh.row_values(rownum))

    dataset.close()
