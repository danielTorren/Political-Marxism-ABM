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

    plt.figure()
    for i in range(numAgent):
        plt.plot(timelist, PropertyData.iloc[i]/Data["Economy_PriceIndex"].iloc[0])
    plt.xlabel(r"Time /years")
    plt.ylabel(r"real %s" % property)
    plt.grid()
    plt.savefig(plotsName + "/"+ property + "_timeseries_real.png" , dpi = 600)

def genericPanelRealloop(axs,PropertyData,PriceIndex,propertyName,numAgent,timelist):
    for i in range(numAgent):
        axs.plot(timelist, PropertyData.iloc[i]/PriceIndex.iloc[0])
    axs.set_xlabel(r"Time /years")
    axs.set_ylabel(r"real %s" % propertyName)
    axs.grid()

def genericPanelloop(axs,PropertyData,propertyName,numAgent,timelist):
    for i in range(numAgent):
        axs.plot(timelist, PropertyData.iloc[i])
    axs.set_xlabel(r"Time /years")
    axs.set_ylabel(r"%s" % propertyName)
    axs.grid()

def panel_of_plots_real(Data,dataList,nrows,ncols,PriceIndex):

    PropertyData = Data[dataList[0]]
    dfShape = PropertyData.shape
    numAgent = dfShape[0]
    numSteps = dfShape[1]
    timelist = range(numSteps)

    fig, axs = plt.subplots(nrows,ncols)

    ax_list = fig.axes

    for i in range(len(dataList)):
        genericPanelRealloop(ax_list[i],Data[dataList[i]],PriceIndex,dataList[i],numAgent,timelist)
    plt.tight_layout()

def panel_of_plots(Data,dataList,nrows,ncols):

    PropertyData = Data[dataList[0]]
    dfShape = PropertyData.shape
    numAgent = dfShape[0]
    numSteps = dfShape[1]
    timelist = range(numSteps)

    fig, axs = plt.subplots(nrows,ncols)

    ax_list = fig.axes

    for i in range(len(dataList)):
        genericPanelloop(ax_list[i],Data[dataList[i]],dataList[i],numAgent,timelist)
    plt.tight_layout()


def producePlots(FILENAME,plotBoolean,plotRealBoolean,):

    dataName = FILENAME + "/Data"
    plotsName =FILENAME + "/Plots"
    #["tenant_Production","tenant_Wealth","tenant_Rent","landlord_Consumption","landlord_Wealth","landlord_TotalRent","landlord_TotalProdSurplus","landlord_NumCustomary","landlord_NumLeasehold","landlord_NumWageLabourer","economy_Inflation" ,"economy_PriceIndex","economy_TotalWealth"]

    loadBoolean = plotBoolean + plotRealBoolean #["tenant_Production","tenant_Wealth","tenant_Rent","landlord_Consumption","landlord_Wealth","landlord_TotalRent","landlord_TotalProdSurplus","landlord_NumCustomary","landlord_NumLeasehold","landlord_NumWageLabourer","economy_Inflation" ,"economy_PriceIndex","economy_TotalWealth"]
    Data = loadData(dataName,loadBoolean)

    for i in plotBoolean:
        standard_timeseries_plot(Data,plotsName,i)

    for i in plotRealBoolean:
        standard_real_timeseries_plot(Data,plotsName,i)

def producePanelPlots(FILENAME,plotList,loadBoolean):

    dataName = FILENAME + "/Data"
    Data = loadData(dataName,loadBoolean)
    PriceIndex = Data["Economy_PriceIndex"]
    for i in plotList:
        if i["Real"]:
            panel_of_plots_real(Data,i["dataList"],i["nrows"],i["ncols"],PriceIndex)
        else:
            panel_of_plots(Data,i["dataList"],i["nrows"],i["ncols"])


if __name__ == "__main__":
    FILENAME = "Results/society_10_50_0.995_1_250_100_10_1000_100"

    #SINGLE PLOTS
    plotBoolean = ["Economy_PriceIndex","Economy_Inflation"]
    plotRealBoolean = ["Economy_TotalWealth","Landlord_Income","Landlord_CapitalInvest"]
    producePlots(FILENAME,plotBoolean,plotRealBoolean)

    #PANEL PLOTS
    loadBoolean = ["Economy_PriceIndex","Tenant_Production","Tenant_Wealth","Tenant_Rent","Tenant_CapitalInvest","Landlord_Consumption","Landlord_Wealth","Landlord_TotalRent","Landlord_TotalProdSurplus","Landlord_NumCustomary","Landlord_NumLeasehold","Landlord_NumWageLabourer","Tenant_Rent"]
    plotList = [
        {"dataList":["Tenant_Production","Tenant_Wealth","Tenant_Rent","Tenant_CapitalInvest"],"nrows":2,"ncols":2 ,"Real":True},
        {"dataList":["Landlord_Consumption","Landlord_Wealth","Landlord_TotalRent","Landlord_TotalProdSurplus"],"nrows":2,"ncols":2,"Real":True },
        {"dataList":["Landlord_NumCustomary","Landlord_NumLeasehold","Landlord_NumWageLabourer","Tenant_Rent"],"nrows":2,"ncols":2,"Real":False },
        
    ]
    producePanelPlots(FILENAME,plotList,loadBoolean)

    plt.show()