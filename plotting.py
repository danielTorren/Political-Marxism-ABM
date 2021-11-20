import matplotlib.pyplot as plt
import pandas as pd
    
def loadData(dataName, loadBoolean):
    data = {}
    ####TENANT DATA
    for i in loadBoolean:
        data[i] = pd.read_csv(dataName + "/" + i +".csv",header=None )

    return data
    

def standard_timeseries_plot(Data,plotsName,property):
    PropertyData = Data[property]
    dfShape = PropertyData.shape
    numAgent = dfShape[0]
    numSteps = dfShape[1]
    timelist = range(numSteps)

    plt.figure()
    for i in range(numAgent):
        plt.plot(timelist, PropertyData.iloc[i])
    plt.xlabel(r"Time /years")
    plt.ylabel(r"%s" % property)
    plt.grid()
    plt.savefig(plotsName + "/"+ property + "_timeseries.png" , dpi = 600)

def standard_real_timeseries_plot(Data,plotsName,property):
    PropertyData = Data[property]
    dfShape = PropertyData.shape
    numAgent = dfShape[0]
    numSteps = dfShape[1]
    timelist = range(numSteps)

    #print("test:",Data["economy_PriceIndex"].iloc[0])
    #print("test2:",PropertyData.iloc[0])
    #print("test3:",PropertyData.iloc[0]/Data["economy_PriceIndex"].iloc[0])
    #quit()

    plt.figure()
    for i in range(numAgent):
        plt.plot(timelist, PropertyData.iloc[i]/Data["Economy_PriceIndex"].iloc[0])
    plt.xlabel(r"Time /years")
    plt.ylabel(r"real %s" % property)
    plt.grid()
    plt.savefig(plotsName + "/"+ property + "_timeseries_real.png" , dpi = 600)

def producePlots(FILENAME,plotBoolean,plotRealBoolean):

    dataName = FILENAME + "/Data"
    plotsName =FILENAME + "/Plots"
    #["tenant_Production","tenant_Wealth","tenant_Rent","landlord_Consumption","landlord_Wealth","landlord_TotalRent","landlord_TotalProdSurplus","landlord_NumCustomary","landlord_NumLeasehold","landlord_NumWageLabourer","economy_Inflation" ,"economy_PriceIndex","economy_TotalWealth"]
    


    loadBoolean = plotBoolean + plotRealBoolean #["tenant_Production","tenant_Wealth","tenant_Rent","landlord_Consumption","landlord_Wealth","landlord_TotalRent","landlord_TotalProdSurplus","landlord_NumCustomary","landlord_NumLeasehold","landlord_NumWageLabourer","economy_Inflation" ,"economy_PriceIndex","economy_TotalWealth"]
    Data = loadData(dataName,loadBoolean)

    for i in plotBoolean:
        standard_timeseries_plot(Data,plotsName,i)

    for i in plotRealBoolean:
        standard_real_timeseries_plot(Data,plotsName,i)

if __name__ == "__main__":
    FILENAME = "Results/society_3_10_0.99_1_100"
    plotBoolean = ["Landlord_NumCustomary","Landlord_NumLeasehold","Landlord_NumWageLabourer","Economy_PriceIndex","Economy_Inflation"]
    plotRealBoolean = ["Tenant_Production","Tenant_Wealth","Tenant_Rent","Tenant_CapitalInvest","Landlord_Consumption","Landlord_Wealth","Landlord_TotalRent","Landlord_TotalProdSurplus","Economy_TotalWealth"]
    producePlots(FILENAME,plotBoolean,plotRealBoolean)
    plt.show()