
from ultralytics import YOLO
from IO_Module.logger import Logger
class YOLOModel:
    def __init__(self,version:str="yolo26n.pt"):
        self.__ver=version
        self.__yolo=YOLO(self.__ver)
        
    @property
    def version(self):
        return self.__ver
    @version.setter
    def setVersion(self,version):
        self.__ver=version
        
        
    def train(self,data,epochs:int=10,**kwargs):
        Logger.hook_stdout()
        Logger.info(f"Model {self.version} is Training...")
        self.__yolo.train(data=data,epochs=epochs,**kwargs)
        return Logger.info('Base Model Training Complete.')
    
    
    
    
    
    
    
if __name__=='__main__':
    test1=YOLOModel()
    print(test1.version)
    import torch
    print(torch.cuda.is_available())
    Logger.info('Operator is conducting a Unit Test Run')
    test1.train(data='data.yaml',
                epochs=10,
                imgsz=480,
                batch=8,     
                device=0,
                workers=6)