from plotting import producePlots
from Run import run
from utility import produceName
import matplotlib.pyplot as plt
import time

"""RUN"""
#####SET PARAMETERS FOR RUN
steps = 500
socNumLandlord = 10
socNumTenantPer = 100
socLandDet = 0.999#0.1% deteriation per year
socInitInflation = 1

tenantWealthMean = 100 
tenantWealthVar = 10
landlordWealthMean = 1000
landlordWealthVar = 100

FILENAME = produceName(socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, steps,tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar )

###RECORD RUN TIME
start_time = time.time()
print("start_time =", time.ctime(time.time()))
run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation , tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar)
print ("Run time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))

"""PLOT"""
start_time = time.time()
plotBoolean = ["Landlord_NumCustomary","Landlord_NumLeasehold","Landlord_NumWageLabourer","Economy_PriceIndex","Economy_Inflation"]
plotRealBoolean = ["Tenant_Production","Tenant_Wealth","Tenant_Rent","Tenant_CapitalInvest","Landlord_Consumption","Landlord_Wealth","Landlord_TotalRent","Landlord_TotalProdSurplus","Economy_TotalWealth"]
producePlots(FILENAME,plotBoolean,plotRealBoolean)
print ("Plot time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))
plt.show()