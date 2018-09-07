from main import RFest
import pandas as pd
import time

generator=RFest()
generator.drop_rate=0.999
generator.rows_range=[500000, 1000000,10000000]
generator.max_depth_range=[10,50,100,200]
generator.inputs_range=[200,300,500,600]
generator.max_features_range=['auto' , 20,50,100]

generator.generate_data()
