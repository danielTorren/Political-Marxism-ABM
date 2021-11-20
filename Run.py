from Economy import Economy
from utility import produceName,saveObjects,saveData,createFolder
import time

def run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar ):

    society = Economy(socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar )

    society.advanceTime()#do one step outside for the sake of inflation
    for i in range(steps-1):#remove one to account for this
        society.adjustInflation()
        society.calcPriceIndex()
        society.advanceTime()
        society.updateHistory()
    
    fileName = produceName(socNumLandlord, socNumTenantPer, socLandDet,  socInitInflation, steps, tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar )
    dataName = createFolder(fileName)
    saveObjects(society, dataName)
    saveData(society, dataName)
    print("File Name:", fileName)

if __name__ == "__main__":
    #####SET PARAMETERS FOR RUN
    steps = 10
    socNumLandlord = 3
    socNumTenantPer = 10
    socLandDet = 0.99#1% inflation????
    socInitInflation = 1
    tenantWealth = 100 
    landlordWealth = 1000

    start_time = time.time()
    print("start_time =", time.ctime(time.time()))
    run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, tenantWealth, landlordWealth )
    print ("time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))