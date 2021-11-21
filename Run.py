from Economy import Economy
from utility import produceName,saveObjects,saveData,createFolder
import time

def run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar ,landlordConsumMean,landlordConsumVar,initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease,rentTimer):

    society = Economy(socNumLandlord, socNumTenantPer, socLandDet, socInitInflation, tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar,landlordConsumMean,landlordConsumVar,initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease ,rentTimer)

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
    #Key
    steps = 300
    socNumLandlord = 5
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

    start_time = time.time()
    print("start_time =", time.ctime(time.time()))
    run(steps,socNumLandlord, socNumTenantPer, socLandDet, socInitInflation , tenantWealthMean,tenantWealthVar, landlordWealthMean, landlordWealthVar,landlordConsumMean,landlordConsumVar,initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease,rentTimer)
    print ("time taken: %s minutes" % ((time.time()-start_time)/60), "or %s s"%((time.time()-start_time)))