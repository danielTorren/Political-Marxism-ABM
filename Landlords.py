from Tenants import Tenant
import numpy as np


class LandLord():
    """ 
        Agent class for landlord, 
        Actions: consume, collect rent,
        change rent type and re-invest surplus into their land

        Porperties: list of tenants (types of rent, productivity, rent to be paid),
        consumption
    """
    def __init__(self,ID,tenantsID, initConsum, initNumTenants, initWealth, Inflation, landList,tenantWealthMean ,tenantWealthVar,improvementVar, initImprovement,initImprovementCost, initImprovementIncrease):
        self.ID = ID
        self.Consumption = initConsum
        self.numTenants = initNumTenants
        self.askOffer = self.Consumption/self.numTenants # split rent evenly
        self.Wealth = initWealth
        self.Inflation = Inflation
        self.numCustomary = self.numTenants
        self.numLeasehold = 0
        self.numWageLabourer = 0
        self.tenancyTypeList = ["Customary","Leasehold", "WageLabourer"]
        self.landList = landList


        #tenant properties
        self.tenantWealthMean = tenantWealthMean
        self.tenantWealthVar = tenantWealthVar
        self.improvementVar = improvementVar
        self.initImprovement = initImprovement
        self.initImprovementCost = initImprovementCost
        self.initImprovementIncrease = initImprovementIncrease
        self.tenantsID = tenantsID

        self.tenantsList = []
        self.populateTenants()
        

        self.totalRent = self.Consumption#START THE MODEL IN A SORT OF STASIS, WHERE THINGS BALENCE OUT
        self.totalProdSurplus = 0

        self.historyConsumption = [self.Consumption]
        self.historyWealth = [self.Wealth]
        self.historyNumTenants = [self.numTenants]
        self.historyTotalRent = [self.totalRent]
        self.historyTotalProdSurplus = [self.totalProdSurplus]
        self.historyNumCustomary = [self.numCustomary]
        self.historyNumLeasehold = [self.numLeasehold]
        self.historyNumWageLabourer = [self.numWageLabourer]
        self.historyAskOffer = [self.askOffer]
        

    def populateTenants(self):
        ConsumVal =  1# set to unity
        wealthVals =  np.random.normal(self.tenantWealthMean, self.tenantWealthVar,self.numTenants)#set to wealt per tenant as mean # self.Wealth/self.numTenants
        TenancyVal = "Customary"

        for i in range(self.numTenants):
            self.tenantsList.append(Tenant(self.tenantsID[i],self.ID ,ConsumVal,wealthVals[i],self.askOffer,TenancyVal,self.Inflation, self.landList[i],self.improvementVar, self.initImprovement,self.initImprovementCost, self.initImprovementIncrease))
        

    def consumInflation(self):
        self.Consumption = self.Consumption*self.Inflation

    def Feed(self):
        self.Wealth -= self.Consumption

    def collectRentandPayWages(self):
        self.totalRent = 0#reset counters
        self.totalProdSurplus = 0#reset counters

        for i in self.tenantsList:
            if i.tenacyType == "WageLabourer":
                prodSurplus = (i.Production - i.Consumption)
                self.Wealth += prodSurplus
                self.totalProdSurplus += prodSurplus
            else:
                rent = i.payRent()
                self.Wealth += rent
                self.totalRent += rent

    def adjustTenancy(self):
        """
        if self.Wealth < 0 and self.numLeasehold > 0:
            rentIncrease = abs(self.Wealth)/self.numLeasehold
            for i in self.tenantsList:
                if i.tenacyType == "Leasehold" :
                    i.Rent += rentIncrease
        """

        for i in self.tenantsList:
            if i.Wealth < 0:
                if i.tenacyType == "Customary":
                    i.leaseholdSwitch()
                    self.numCustomary-=1
                    self.numLeasehold+=1
                elif i.tenacyType == "Leasehold":
                    i.wageLabouerSwitch()
                    self.numLeasehold-=1
                    self.numWageLabourer+=1

    def reinvestCapital(self):
        if self.Wealth > 0 and self.numWageLabourer > 0:
            increase = self.Wealth/self.numWageLabourer
            for i in self.tenantsList:
                if i.tenacyType == "WageLabourer":
                    self.Wealth -= i.improvementCost
                    i.improvement += i.improvementIncrease*np.random.normal(1, self.improvementVar)#arbitrary
                    
            
    def updateHistory(self):
        self.historyConsumption.append(self.Consumption)
        self.historyWealth.append(self.Wealth) 
        self.historyNumTenants.append(self.numTenants) 
        self.historyTotalRent.append(self.totalRent)
        self.historyTotalProdSurplus.append(self.totalProdSurplus)
        self.historyNumCustomary.append(self.numCustomary)
        self.historyNumLeasehold.append(self.numLeasehold)
        self.historyNumWageLabourer.append(self.numWageLabourer)
        self.historyAskOffer.append(self.askOffer)

    def advanceTime(self):
        for i in self.tenantsList:
            i.Inflation = self.Inflation
            i.advanceTime()

        self.consumInflation()
        self.Feed()
        self.collectRentandPayWages()
        self.adjustTenancy()
        self.reinvestCapital()
        self.updateHistory()