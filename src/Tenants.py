import numpy as np
class Tenant():
    """ 
        Agent class for tenants
        Actions: consume, exploit land, pay rent, sell surplus
        Properties: lease length, rent cost, land productivity, consumption,  landlordID, ID, inflation
    """
    def __init__(self, ID,landlordID ,initConsum, initWealth, initRent, initTenancy, Inflation, Land, improvementVar, initImprovement, initImprovementCost, initImprovementIncrease,rentTimer):#,landlordID, ID,
        self.ID = ID
        self.landlordID = landlordID
        self.Consumption = initConsum
        self.Wealth = initWealth

        self.Rent = initRent
        self.rentBid = self.Rent*np.random.normal(1, 0.01)#just to add slightly different rent bid prices
        self.rentTimer = rentTimer
        self.tenacyType = initTenancy

        self.improvementVar = improvementVar
        self.improvement = initImprovement
        self.improvementCost = initImprovementCost
        self.improvementIncrease = initImprovementIncrease

        self.Inflation = Inflation
        self.capitalInvest = 0
        self.Land = Land

        self.Production = self.Land.returnLandProduct()

        self.historyProduction = [self.Production]
        self.historyConsumption = [self.Consumption]
        self.historyWealth = [self.Wealth ]
        self.historyRent = [self.Rent]
        self.historyTenacyType = [self.tenacyType]
        self.historyCapitalInvest = [self.capitalInvest]
        self.historyRentBid = [self.rentBid]
    
    def adjustConsumption(self):
        self.Consumption = self.Consumption*self.Inflation

    def adjustImprovementCost(self):
        self.improvementCost = self.improvementCost*self.Inflation

    def Feed(self):
        self.Wealth -= self.Consumption

    def Produce(self):
        self.Wealth += self.Land.landProduct

    def updateProduction(self):
        self.Production = self.Land.returnLandProduct()

    def payRent(self):
        self.Wealth -= self.Rent
        return self.Rent

    def reinvestCapital(self):
        self.Wealth -= self.improvementCost
        self.capitalInvest += self.improvementCost
        self.improvement += self.improvementIncrease*np.random.normal(1, self.improvementVar)#arbitrary

    def leaseholdSwitch(self):
        self.tenacyType = "Leasehold"
        

    def wageLabouerSwitch(self):
        self.tenacyType = "WageLabourer" #wage labourers no longer consume or have welath as they are paid the minimum
        self.Wealth =  np.nan
        self.Rent =  np.nan
        self.capitalInvest =  np.nan

    def updateHistory(self):
        self.historyProduction.append(self.Production)
        self.historyConsumption.append(self.Consumption)
        self.historyWealth.append(self.Wealth) 
        self.historyRent.append(self.Rent) 
        self.historyTenacyType.append(self.tenacyType)
        self.historyCapitalInvest.append(self.capitalInvest)
        self.historyRentBid.append(self.rentBid)

    def advanceTime(self):

        self.adjustConsumption()
        self.adjustImprovementCost()
        
        self.Land.updatelandProduct(self.improvement)
        self.updateProduction()

        if self.tenacyType == "Customary" or "Leasehold":
            self.Feed()
            self.Produce()
            
            if self.tenacyType == "Leasehold":
                self.rentTimer -= 1
                if self.Wealth > self.improvementCost:
                    self.reinvestCapital()
                
                
        self.updateHistory()