from main import RFest
import pandas as pd
import time

generator=RFest()
generator.drop_rate=0.999
generator.rows_range=[25000000, 50000000,100000000]
generator.max_depth_range=[10,50,100,200]
generator.inputs_range=[200,500,700,1000]
generator.max_features_range=['auto' , 20,50,100]

generator.generate_data()
