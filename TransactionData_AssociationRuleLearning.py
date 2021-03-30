### Analyze data ###

## Association Rule Learning ##

import pandas as pd
import numpy as np
from apyori import apriori


products2 = pd.read_csv('output1_c.csv', header = None, keep_default_na = False)

products2

items2 = []
for i in range(len(products2.axes[0])):
    items2.append([str(products2.values[i,j]) for j in range(0, len(products2.columns))])


rules2 = apriori(transactions = items2, min_support = 0.01, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

results2 = list(rules2)
results2


## Putting the results well organised into a Pandas DataFrame
def inspect(results2):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame2 = pd.DataFrame(inspect(results2), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
resultsinDataFrame2

## Displaying the results sorted by descending Support, Confidence and lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Support')

resultsinDataFrame.nlargest(n = 10, columns = 'Confidence')

resultsinDataFrame.nlargest(n = 10, columns = 'Lift')




