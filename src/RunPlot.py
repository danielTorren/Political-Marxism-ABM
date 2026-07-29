from plotting import producePlots,producePanelPlots
from Run import run
from utility import produceName
import matplotlib.pyplot as plt
import time

"""RUN"""
#####SET PARAMETERS FOR RUN
#Key
steps = 300
socNumLandlord = 10
socNumTenantPer = 50
initLandProduct = 2#jack up the inital productivity of land
socLandDet = 0.995#0.5% deteriation per year
rentTimer = 10

#economy properties
socInitInflation = 1

#landlord proerties
landlordConsumMean = socNumTenantPer
landlordConsumVar = 10
landlordWealthMean = 1000
landlordWealthVar = 100

#tenant properties
tenantWealthMean = 100 
tenantWealthVar = 10

improvementVar = 1 #mean of 1
initImprovement = 1
initImprovementCost = 1  
initImprovementIncrease = 0.02

#environmental properties


FILENAME = produceName(socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, steps,tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar )

###RECORD RUN TIME
start_time = time.time()
print("start_time =", time.ctime(time.time()))
run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation , tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar,landlordConsumMean,landlordConsumVar,initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease,rentTimer)
print ("Run time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

"""PLOT"""
start_time = time.time()
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

print("Plot time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
plt.show()