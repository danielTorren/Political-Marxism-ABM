import numpy as np
from statistics import mean
from Landlords import LandLord
from Land import Land

class Economy():
    """ 
        Contains all the things
        properties: total value , inflation, agents, land

    """
    def __init__(self,numLandlord, numTenantPer, landDet, Inflation, tenantWealthMean,tenantWealthVar, landlordWealthMean,landlordWealthVar,landlordConsumMean, landlordConsumVar, initLandProduct, improvementVar, initImprovement,initImprovementCost, initImprovementIncrease,rentTimer):

        #environmental properties
        self.landDet = landDet
        self.Inflation = Inflation
        self.PriceIndex = 1
        self.initLandProduct = initLandProduct

        #landlord properties
        self.numLandlord = numLandlord
        self.numTenantPer = numTenantPer
        self.landlordConsumMean = landlordConsumMean
        self.landlordConsumVar = landlordConsumVar
        self.landlordWealthMean = landlordWealthMean
        self.landlordWealthVar = landlordWealthVar

        #tenant properties
        self.tenantWealthMean = tenantWealthMean
        self.tenantWealthVar = tenantWealthVar
        self.improvementVar = improvementVar
        self.initImprovement = initImprovement
        self.initImprovementCost = initImprovementCost
        self.initImprovementIncrease = initImprovementIncrease
        self.rentTimer = rentTimer
        
        #create economy
        self.LandList = []
        self.LandLordList = []#list of all landlord objects
        self.populateLandList()
        self.populateLandlordlist()
        self.TenantList = []#list of all tenant objects
        self.createTenantList()

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
            self.LandLordList.append(LandLord(i,range(i,i+self.numTenantPer),consumptionVals[i],self.numTenantPer,initWealthVals[i] ,self.Inflation, landLordLand,self.tenantWealthMean,self.tenantWealthVar,self.improvementVar, self.initImprovement,self.initImprovementCost, self.initImprovementIncrease ,self.rentTimer))

    def createTenantList(self):
        for i in range(len(self.LandLordList)):
                for v in range(len(self.LandLordList[i].tenantsList)):
                        self.TenantList.append(self.LandLordList[i].tenantsList[v])


    def calculateTotalWealth(self):
        totalWealth = 0
        for i in self.TenantList:
                totalWealth += i.Production

        return totalWealth
    
    def adjustInflation(self):
        totalWealth = self.calculateTotalWealth()
        self.Inflation = totalWealth/self.totalWealth
        self.totalWealth = totalWealth
    
    def calcPriceIndex(self):
        self.PriceIndex = self.PriceIndex*self.Inflation

    def changeTenantLandlord(self,tenant,oldLandlord,NewLandlord):
        NewLandlord.tenantsList.append(tenant)#add tenant to new landlord list
        oldLandlord.tenantsList.remove(tenant)#remove tenant from old land lord list#THIS IS VERY DUBIOUS
        tenant.landlordID = NewLandlord.ID#change tenant landlord ID

    def rentalMarket(self):

        """currently the agets arent swapping landlord which they really should! but how do you keep track of this???"""
        buyersOffer = []
        sellerAsk = []

        for i in range(len(self.LandLordList)):
            if self.LandLordList[i].numLeasehold > 0:
                for v in range(len(self.LandLordList[i].tenantsList)):
                    #print("rent timer",self.LandLordList[i].tenantsList[v].rentTimer)
                    if self.LandLordList[i].tenantsList[v].tenacyType == "Leasehold" and self.LandLordList[i].tenantsList[v].rentTimer == 0:
                        self.LandLordList[i].tenantsList[v].rentTimer = self.rentTimer#reset rent timer
                        buyersOffer.append(self.LandLordList[i].tenantsList[v])
                        sellerAsk.append(self.LandLordList[i])
        
        buyersOffer.sort(reverse=True, key = lambda x: x.rentBid)
        sellerAsk.sort(reverse=False, key = lambda x: x.askOffer)
        #if len(buyersOffer) > 2:
        #    print("buyersOffer v sellerAsk: ",buyersOffer[0].rentBid,buyersOffer[-1].rentBid ,sellerAsk[0].askOffer,sellerAsk[-1].askOffer)
        #successfull offers
        latestSuccessfullOfferAsk = [0,0]
        for i in range(len(buyersOffer)):
            if(buyersOffer[i].rentBid >= sellerAsk[i].askOffer):
            #change the rent of buyer
                latestSuccessfullOfferAsk[0],latestSuccessfullOfferAsk[1] = buyersOffer[i].rentBid,sellerAsk[i].askOffer#keep track of latest deal
                buyersOffer[i].Rent = buyersOffer[i].rentBid
            #change opinion of buyer
                buyersOffer[i].rentBid = buyersOffer[i].Rent - ((buyersOffer[i].Rent-sellerAsk[i].askOffer) /2)#change to be difference
            #change buyer of landlord and change tenants of seller
                if(buyersOffer[i].landlordID !=sellerAsk[i].ID):#if they have different IDs
                    oldlandlord = [x for x in self.LandLordList if x.ID == buyersOffer[i].landlordID]
                    #print("change landlord:" ,buyersOffer[i][1],oldlandlord[0],sellerAsk[i][1])
                    self.changeTenantLandlord(buyersOffer[i],oldlandlord[0],sellerAsk[i])
            else:
                break
        
        clearingPrice = (latestSuccessfullOfferAsk[0]+latestSuccessfullOfferAsk[1])/2

        for i in range(len(buyersOffer)):
            #change the rent of buyer
            buyersOffer[i].Rent = clearingPrice
            #change opinion of buyer
            buyersOffer[i].rentBid = buyersOffer[i].Rent + ((clearingPrice - buyersOffer[i].rentBid)/2)#change to be difference
            #change buyer of landlord and change tenants of seller
            oldlandlord = [x for x in self.LandLordList if x.ID == buyersOffer[i].landlordID]
            self.changeTenantLandlord(buyersOffer[i],oldlandlord[0],sellerAsk[i])
            
        #change opinion of seller
        for i in range(len(self.LandLordList)):
            if self.LandLordList[i].numLeasehold > 0:
                rentLandlord = [x.Rent for x in self.LandLordList[i].tenantsList if x.tenacyType == "Leasehold"]
                self.LandLordList[i].askOffer -= (self.LandLordList[i].askOffer -  mean(rentLandlord))/2#adjust the asking rent based off of the average of current rent#wont this just tend towards one value, need to introduce an incentive to icnrease rent!

    def updateHistory(self):
        self.historyInflation.append(self.Inflation)
        self.historyPriceIndex.append(self.PriceIndex)
        self.historyTotalWealth.append(self.totalWealth)
        

    def advanceTime(self):

        for i in self.LandLordList:
            i.Inflation = self.Inflation
            i.advanceTime()
        
        self.rentalMarket()