import csv

def getListOfGames(regime):
    fileName = "./files/"
    fileName += "sonic-train.csv" if regime == "train" else "sonic-validation.csv"

    names = list()

    with open(fileName, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        firstline = True
        for row in spamreader:
            if firstline:    #skip first line
                firstline = False
                continue
            names.append(row[0])
    return names
