import numpy as np
from statistics import mean
from Landlords import LandLord
from Land import Land

class Economy():
    """ 
        Contains all the things
        properties: total value , inflation, agents, land

    """
    def __init__(self,numLandlord, numTenantPer, landDet, Inflation, tenantWealthMean,tenantWealthVar, landlordWealthMean,landlordWealthVar,landlordConsumMean, landlordConsumVar, initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease):

        self.numLandlord = numLandlord
        self.numTenantPer = numTenantPer
        self.landDet = landDet
        self.Inflation = Inflation
        self.PriceIndex = 1
        self.initLandProduct = initLandProduct

        self.tenantWealthMean = tenantWealthMean
        self.tenantWealthVar = tenantWealthVar
        self.landlordConsumMean = landlordConsumMean
        self.landlordConsumVar = landlordConsumVar
        self.landlordWealthMean = landlordWealthMean
        self.landlordWealthVar = landlordWealthVar

        #tenant properties
        self.improvementVar = improvementVar
        self.initImprovement = initImprovement
        self.initImprovementCost = initImprovementCost
        self.initImprovementIncrease = initImprovementIncrease
        
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
        consumptionVals = np.random.normal(self.landlordConsumMean, self.landlordConsumVar,self.numLandlord)#set as the number of tenants so rent per tenant is close to unity
        initWealthVals =  np.random.normal(self.landlordWealthMean, self.landlordWealthVar,self.numLandlord)#set relative to number of tenants per, in this case 2 #2*self.numTenantPer

        for i in range(self.numLandlord):#ID,landDet, initConsum, initNumTenants, initWealth, Inflation )
            landLordLand = self.LandList[i:i+self.numTenantPer]
            self.LandLordList.append(LandLord(i,range(i,i+self.numTenantPer),consumptionVals[i],self.numTenantPer,initWealthVals[i] ,self.Inflation, landLordLand,self.tenantWealthMean,self.tenantWealthVar,self.improvementVar, self.initImprovement,self.initImprovementCost, self.initImprovementIncrease ))

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

    def rentalMarket(self):

        """currently the agets arent swapping landlord which they really should! but how do you keep track of this???"""
        buyersOffer = []
        sellerAsk = []

        for i in range(len(self.LandLordList)):
            if self.LandLordList[i].numLeasehold > 0:
                for v in range(len(self.LandLordList[i].tenantsList)):
                    if self.LandLordList[i].tenantsList[v].tenacyType == "Leasehold":
                        buyersOffer.append([self.LandLordList[i].tenantsList[v].rentBid,self.LandLordList[i].tenantsList[v]])
                        sellerAsk.append([self.LandLordList[i].askOffer,self.LandLordList[i]])
        
        buyersOffer.sort(reverse=True, key = lambda x: x[0])
        sellerAsk.sort(reverse=False, key = lambda x: x[0])

        #successfull offers
        latestSuccessfullOfferAsk = [0,0]
        for i in range(len(buyersOffer)):
            if(buyersOffer[i][0] >= sellerAsk[i][0]):#set the rent of buyer
                latestSuccessfullOfferAsk[0],latestSuccessfullOfferAsk[1] = buyersOffer[i][0],sellerAsk[i][0]
                buyersOffer[i][1].Rent = buyersOffer[i][0]
            #change opinion of buyer
                buyersOffer[i][1].rentBid = buyersOffer[i][1].Rent - ((buyersOffer[i][0]-sellerAsk[i][0]) /2)#change to be difference
            #remove buyer    
                buyersOffer[i].pop(0)
            else:
                break
        
        clearingPrice = (latestSuccessfullOfferAsk[0]+latestSuccessfullOfferAsk[1])/2
        for i in range(len(buyersOffer)):
                buyersOffer[i][1].Rent = clearingPrice
                #change opinion of buyer
                buyersOffer[i][1].rentBid = buyersOffer[i][1].Rent + ((clearingPrice - buyersOffer[i][0])/2)#change to be difference
            
        #change opinion of seller
        for i in range(len(self.LandLordList)):
            if self.LandLordList[i].numLeasehold > 0:
                rentLandlord = [x.Rent for x in self.LandLordList[i].tenantsList if x.tenacyType == "Leasehold"]
                self.LandLordList[i].askOffer -= (self.LandLordList[i].askOffer -  mean(rentLandlord))/2

    def updateHistory(self):
        self.historyInflation.append(self.Inflation)
        self.historyPriceIndex.append(self.PriceIndex)
        self.historyTotalWealth.append(self.totalWealth)
        

    def advanceTime(self):

        for i in self.LandLordList:
            i.Inflation = self.Inflation
            i.advanceTime()
        
        self.rentalMarket()