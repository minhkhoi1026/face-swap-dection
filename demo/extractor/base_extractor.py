import abc

class BaseExtractor(object):
    @abc.abstractmethod
    def extract(self, data):
        raise NotImplementedError("Should have implemented this")
