from plotting import producePlots
from Run import run
from utility import produceName
import matplotlib.pyplot as plt
import time

"""RUN"""
#####SET PARAMETERS FOR RUN
steps = 10

#economy properties
socNumLandlord = 3
socNumTenantPer = 10
socInitInflation = 1

#landlord proerties
landlordConsumMean = socNumTenantPer
landlordConsumVar = 10
landlordWealthMean = 1000
landlordWealthVar = 100

#tenant properties
tenantWealthMean = 100 
tenantWealthVar = 10
initLandProduct = 1
improvementVar = 0.1 #mean of 1
initImprovement = 1
initImprovementCost = 1  
initImprovementIncrease = 0.002

#environmental properties
socLandDet = 0.999#0.1% deteriation per year
FILENAME = produceName(socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, steps,tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar )

###RECORD RUN TIME
start_time = time.time()
print("start_time =", time.ctime(time.time()))
run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation , tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar,landlordConsumMean,landlordConsumVar,initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease)
print ("Run time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

"""PLOT"""
start_time = time.time()
plotBoolean = ["Landlord_NumCustomary","Landlord_NumLeasehold","Landlord_NumWageLabourer","Economy_PriceIndex","Economy_Inflation"]
plotRealBoolean = ["Tenant_Production","Tenant_Wealth","Tenant_Rent","Tenant_CapitalInvest","Landlord_Consumption","Landlord_Wealth","Landlord_TotalRent","Landlord_TotalProdSurplus","Landlord_AskOffer","Economy_TotalWealth"]
producePlots(FILENAME,plotBoolean,plotRealBoolean)
print ("Plot time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
plt.show()