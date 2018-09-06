from main import RFest
import pandas as pd
import time

generator=RFest()
generator.drop_rate=0.999
generator.rows_range=[100000, 1000000, 10000000, 25000000, 50000000]
generator.max_depth_range=[10,50,100,200]
generator.inputs_range=[10,50,100,1000]
generator.max_features_range=['auto'  ,10,20,50]

generator.generate_data()
