import numpy as np
import pandas as pd

class Logger:
    def __init__(self) -> None:
        # Store (x,y,z,t, confidence, class) in here
        self.records = np.empty((0,6))


    def record_value(self, record):
        # The caller must make sure the dimensions match
        self.records = np.concatenate((self.records, record))
    
    def export_to_csv(self, output_location):
        cols = ['x', 'y', 'z', 't', 'confidence', 'class']
        df = pd.DataFrame(data=self.records, columns=cols)
        df.to_csv(output_location)

    