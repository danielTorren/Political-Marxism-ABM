import pickle
import csv
import os

def produceName(socNumLandlord, socNumTenantPer, socLandDet, socInflation, steps, tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar):
    runName = "society_" + str(socNumLandlord) + "_" + str(socNumTenantPer) + "_" + str(socLandDet) + "_" + str(socInflation) + "_" + str(steps) + "_" + str(tenantWealthMean) + "_" + str(tenantWealthVar) + "_" + str(landlordWealthMean) + "_" + str(landlordWealthVar)
    fileName = "Results/"+ runName
    return fileName

def saveObjects(data,name):
    name
    with open(name +'.pkl', 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def loadObjects(name):
    with open(name +'.pkl', 'rb') as inp:
        data = pickle.load(inp)
    return data

def saveDataDict(flatData,dataSaveList):
    dataDict = {}

    for i in dataSaveList:
        dataDict[i] = []

    for i in range(len(flatData)):
        for v in dataSaveList:
            dataDict[v].append(eval("flatData[i].history" + v ))#work out variable name and append this data usign eval to ge the variable
    return dataDict

def saveCSV(dataName,endName,dataSave):
    with open(dataName + endName, 'w', newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(dataSave)

def saveFromList(dataName,dataSaveList,dataDict,agentClass):
    for i in dataSaveList:
        endName = "/" + agentClass + "_" + i + ".csv"  #'/Tenant_LandDet.csv'
        saveCSV(dataName,endName,dataDict[i])

def saveData(data, dataName):
    
    """save data from tenants"""
    flatTenantData = []
    for i in range(len(data.LandLordList)):
        for v in range(len(data.LandLordList[i].tenantsList)):
            flatTenantData.append(data.LandLordList[i].tenantsList[v])

    #list of things to be saved
    dataSaveTenantList = ["LandDet","Production","Consumption","Wealth","Rent","TenacyType","CapitalInvest"]
    #create dict with flatdata
    dataTenantDict = saveDataDict(flatTenantData,dataSaveTenantList)
    #save as CSV
    saveFromList(dataName,dataSaveTenantList,dataTenantDict,"Tenant")

    """save data from landlords"""
    #list of things to be saved
    dataSaveLandlordList = ["Consumption","Wealth","NumTenants","Wealth","TotalRent","TotalProdSurplus","NumCustomary","NumLeasehold","NumWageLabourer"]
    #create dict with flatdata
    dataLandlordDict = saveDataDict(data.LandLordList,dataSaveLandlordList)
    #save as CSV
    saveFromList(dataName,dataSaveLandlordList,dataLandlordDict,"Landlord")

    """SAVE DATA FROM ECONOMY"""
    #list of things to be saved
    dataSaveEconomyList = ["Inflation","PriceIndex","TotalWealth"]
    #create dict with flatdata
    dataEconomyDict = saveDataDict([data],dataSaveEconomyList)
    #save as CSV
    saveFromList(dataName,dataSaveEconomyList,dataEconomyDict,"Economy")

def createFolder(fileName):
    #check for resutls folder
    if str(os.path.exists("Results")) == "False":
        os.mkdir("Results")

    #check for runName folder
    if str(os.path.exists(fileName)) == "False":
        os.mkdir(fileName)

    #make data folder:#
    dataName = fileName +"/Data"
    if str(os.path.exists(dataName)) == "False":
        os.mkdir(dataName)
    #make plots folder:
    plotsName = fileName +"/Plots"
    if str(os.path.exists(plotsName)) == "False":
        os.mkdir(plotsName)

    return dataName