import numpy as np

from Landlords import LandLord
from Land import Land

class Economy():
    """ 
        Contains all the things
        properties: total value , inflation, agents, land

    """
    def __init__(self,numLandlord, numTenantPer, landDet, Inflation, tenantWealthMean,tenantWealthVar, landlordWealthMean,landlordWealthVar, initLandProduct = 1):

        self.numLandlord = numLandlord
        self.numTenantPer = numTenantPer
        self.landDet = landDet
        self.Inflation = Inflation
        self.PriceIndex = 1
        self.initLandProduct = initLandProduct

        self.tenantWealthMean = tenantWealthMean
        self.tenantWealthVar = tenantWealthVar
        self.landlordWealthMean = landlordWealthMean
        self.landlordWealthVar = landlordWealthVar
        
        self.LandList = []
        self.LandLordList = []
        self.populateLandList()
        self.populateLandlordlist()
        self.totalWealth = self.calculateTotalWealth()

        self.historyInflation = [self.Inflation]
        self.historyPriceIndex = [self.PriceIndex]
        self.historyTotalWealth = [self.totalWealth]

    def populateLandList(self):
        for i in range(self.numLandlord*self.numTenantPer):
            self.LandList.append(Land(i, self.initLandProduct, self.landDet))

    def populateLandlordlist(self):
        consumptionVals = np.random.normal(self.numTenantPer, 0.5,self.numLandlord)#set as the number of tenants so rent per tenant is close to unity
        initWealthVals =  np.random.normal(self.landlordWealthMean, self.landlordWealthVar,self.numLandlord)#set relative to number of tenants per, in this case 2 #2*self.numTenantPer

        for i in range(self.numLandlord):#ID,landDet, initConsum, initNumTenants, initWealth, Inflation )
            landLordLand = self.LandList[i:i+self.numTenantPer]
            self.LandLordList.append(LandLord(i,self.landDet,consumptionVals[i],self.numTenantPer,initWealthVals[i] ,self.Inflation, landLordLand,self.tenantWealthMean,self.tenantWealthVar ))

    def calculateTotalWealth(self):
        totalWealth = 0
        for i in self.LandLordList:
            #totalWealth += i.Wealth
            for v in i.tenantsList:
                #totalWealth += (v.Wealth + v.Production)
                #print("idividual agent production:",v.Production)
                totalWealth += v.Production

        return totalWealth
    
    def adjustInflation(self):
        totalWealth = self.calculateTotalWealth()
        #print("inflation:",totalWealth,self.totalWealth)
        self.Inflation = totalWealth/self.totalWealth
        self.totalWealth = totalWealth
    
    def calcPriceIndex(self):
        self.PriceIndex = self.PriceIndex*self.Inflation

    def updateHistory(self):
        self.historyInflation.append(self.Inflation)
        self.historyPriceIndex.append(self.PriceIndex)
        self.historyTotalWealth.append(self.totalWealth)
        

    def advanceTime(self):
        for i in self.LandLordList:
            i.Inflation = self.Inflation
            i.advanceTime()