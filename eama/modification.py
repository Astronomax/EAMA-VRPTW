from abc import ABC, abstractmethod

class Modification(ABC):
    @abstractmethod
    def penalty_delta(self, alpha, beta):
        pass
    
    @abstractmethod
    def distance_delta(self):
        pass

    @abstractmethod
    def appliable(self, alpha, beta):
        pass

    @abstractmethod
    def apply(self):
        pass