from main import RFest
import pandas as pd
import time

generator=RFest()
generator.drop_rate=0.999
generator.rows_range=[100000, 1000000, 10000000, 25000000, 50000000]

result=generator.generate_data()

result[0].to_csv('input'+str(time.time())+'.csv')
result[1].to_csv('output'+str(time.time())+'.csv')
