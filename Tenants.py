import numpy as np
class Tenant():
    """ 
        Agent class for tenants
        Actions: consume, exploit land, pay rent, sell surplus
        Properties: lease length, rent cost, land productivity, consumption,  landlordID, ID, inflation
    """
    def __init__(self, landDet ,initProd,initConsum, initWealth, initRent, initTenancy, Inflation, Land, initImprovement = 1,initImprovementCost = 1, initImprovementIncrease = 0.002):#,landlordID, ID,
        self.landDet = landDet
        
        self.Consumption = initConsum
        self.Wealth = initWealth
        self.Rent = initRent
        self.tenacyType = initTenancy

        self.improvement = initImprovement
        self.improvementCost = initImprovementCost
        self.improvementIncrease = initImprovementIncrease

        self.Inflation = Inflation
        self.capitalInvest = 0
        self.Land = Land

        self.Production = self.Land.returnLandProduct()

        self.historyLandDet = [self.landDet]
        self.historyProduction = [self.Production]
        self.historyConsumption = [self.Consumption]
        self.historyWealth = [self.Wealth ]
        self.historyRent = [self.Rent]
        self.historyTenacyType = [self.tenacyType]
        self.historyCapitalInvest = [self.capitalInvest]

    #def landDeteriation(self):
    #    self.Production = self.Production*self.landDet
    
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
        """
        increase = self.Production - (self.Inflation - 1)*(self.Rent + self.Consumption)#what extra is needed for next year

        if self.Wealth > increase:
            self.capitalInvest = increase
        else: 
            self.capitalInvest = self.Wealth
        """

        self.Wealth -= self.improvementCost
        self.improvement += self.improvementIncrease*np.random.normal(1, 0.1)#arbitrary
        
    def leaseholdSwitch(self):
        self.tenacyType = "Leasehold"
        

    def wageLabouerSwitch(self):
        self.tenacyType = "WageLabourer" #wage labourers no longer consume or have welath as they are paid the minimum
        self.Wealth = 0
        self.Rent = 0
        self.capitalInvest = 0

    def updateHistory(self):
        self.historyLandDet.append(self.landDet)
        self.historyProduction.append(self.Production)
        self.historyConsumption.append(self.Consumption)
        self.historyWealth.append(self.Wealth) 
        self.historyRent.append(self.Rent) 
        self.historyTenacyType.append(self.tenacyType)
        self.historyCapitalInvest.append(self.capitalInvest)

    def advanceTime(self):
        
        #self.landDeteriation()
        self.adjustConsumption()
        self.adjustImprovementCost()
        
        self.Land.updatelandProduct(self.improvement)
        self.updateProduction()

        if self.tenacyType == "Customary" or "Leasehold":
            self.Feed()
            self.Produce()
            
            if self.tenacyType == "Leasehold" and self.Wealth > 0:
                self.reinvestCapital()
                
        self.updateHistory()