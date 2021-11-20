class Land():
    """ 
        Contains 1 land
        properties: 

    """
    def __init__(self, landID,initLandProduct, initlandProductDet):
        self.landProduct = initLandProduct
        self.landProductDet = initlandProductDet
        self.landID = landID

    def updatelandProduct(self, improvement):
        self.landProduct = self.landProduct*(improvement*self.landProductDet)
    
    def returnLandProduct(self):
        return self.landProduct


        