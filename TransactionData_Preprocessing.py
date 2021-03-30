### Preprocess the dataset ###

## Importing the libraries ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset and preprocessing ##
dataset = pd.read_csv('orders_2019to2021.csv', keep_default_na=False)

dataset.columns = dataset.columns.str.replace(' ', '_').str.replace('(','').str.replace(')','') # replacing the empty space in column names with '_' and get rid of '()'


def preprocess_item_name(item_name):
    item_name = item_name.split(" - ")[0]
    return item_name

def preprocess_int(a):
    if a == "":
        return 0
    
    return int(a)

def preprocess_float(a):
    if a == "":
        return 0.0
    
    return float(a)


List1_int = ["Quantity","Item_#"]
List2_float = ["Discount_Amount","Order_Total_Amount","Order_Subtotal_Amount","Cart_Discount_Amount","Order_Shipping_Amount","Order_Total_Tax_Amount","Item_Cost"]

for x in List1_int:
    dataset[x] = dataset[x].transform(preprocess_int)

    
for x in List2_float:
    dataset[x] = dataset[x].transform(preprocess_float)
    

## Group by Order Number (combined the products bought in the same orders together)##

## Group the purchase items for the same order together
dataset["Item_Name"] = dataset.groupby(dataset['Order_Number'])['Item_Name'].transform(lambda x: ','.join(x))


dataset["SKU"] = dataset.groupby(dataset['Order_Number'])['SKU'].transform(lambda x: '-'.join(x))

## Get the total number of different items 
dataset["Item_#"] = dataset.groupby(dataset['Order_Number'])['Item_#'].transform('max')

## Get the total quantity for each order
dataset["Quantity"] = dataset.groupby(dataset['Order_Number'])['Quantity'].transform('sum')


## Drop duplicated orders, making one row stand for one order
dataset = dataset.drop_duplicates(subset='Order_Number', keep="first",ignore_index='True')


## create a new variable call discount rate for each order
dataset['Discount_Rate'] = round(100*(dataset['Cart_Discount_Amount']/dataset['Order_Subtotal_Amount']),2)

## check if there is any order with 0 dollar for Order_Subtotal_Amount, which indicate invalid order
print(dataset[dataset['Order_Subtotal_Amount'] == 0][['Order_Number','Order_Subtotal_Amount','Cart_Discount_Amount','Payment_Method_Title','Coupon_Code']])

## delete those rows with 0 for Order_Subtotal_Amount
dataset = dataset[dataset['Order_Subtotal_Amount'] != 0]

## Re-categorieze some categorical variables, aggregate those low frequencies to "other" ##

## make the upper and lower case consistant so the same shipping method can be treated as the same no matter using lower or upper case
dataset['Shipping_Method_Title'] = dataset['Shipping_Method_Title'].str.lower()


print(dataset['Shipping_Method_Title'].value_counts()) 

series = pd.value_counts(dataset['Shipping_Method_Title'])
mask = (series/series.sum() * 100) ## get the percentage for each category
print(mask)

mask = (series/series.sum() * 100).lt(2) ## lt(2) differenciate those percentange less than 2
print(mask)
dataset['Shipping_Method_Title'] = np.where(dataset['Shipping_Method_Title'].isin(series[mask].index),'Other',dataset['Shipping_Method_Title'])
## replace those frequency less than 2 (basically besides 'free shipping' and 'flat rate') with "other"

print(dataset['Shipping_Method_Title'].value_counts()) ## check after replacing


dataset['FreeShip_Percent'] = np.where(dataset['Order_Shipping_Amount']==0,0,1)

## Re-categorize state code variable ##
series_state = pd.value_counts(dataset['State_Code_Shipping'])
mask_state = (series_state/series_state.sum() * 100).to_dict()

dataset['Frequency'] = [mask_state[d] for d in dataset.State_Code_Shipping] ## create a new column for frequency of state codes

dataset.loc[dataset['Frequency'] < 4, ['State_Code_Shipping']] = 'Other'

dataset['State_Code_Shipping'].value_counts() ## check after replacing

series_state_2 = pd.value_counts(dataset['State_Code_Shipping'])

mask_state_2 = (series_state_2/series_state_2.sum() * 100)
print(mask_state_2)

## From the results, grouping all other low frequency state results in 55% of all orders, which is the largest group. So the state variable is not quite useful. 


## Re-categorize country code variable ##

series_country = pd.value_counts(dataset['Country_Code_Shipping'])
mask_country = (series_country/series_country.sum() * 100)
print(mask_country)

mask_country = (series_country/series_country.sum() * 100).to_dict() ## get the percentage for each category

dataset['Frequency2'] = [mask_country[i] for i in dataset.Country_Code_Shipping]

## After country "AE" with frequency of 0.2931%, the following ones all less than 0.1%, so group them into "Other" 
dataset.loc[dataset['Frequency2'] < 0.29, ['Country_Code_Shipping']] = 'Other' 


print(dataset.Country_Code_Shipping.value_counts()) ## checking after transform


## re-categorize the column 'coupon_code' to fewer groups: none, coupon, giftcard ##
dataset.Coupon_Code.value_counts().to_dict() ## check all the values in the "Coupon_Code"
dataset['Coupon_Code'] = dataset['Coupon_Code'].replace('', 'none')
dataset['Coupon_Code'] = dataset['Coupon_Code'].replace('wc.*', 'giftcard', regex=True)
dataset.loc[~dataset["Coupon_Code"].isin(['none','giftcard']), "Coupon_Code"] = "coupon"
dataset.Coupon_Code.value_counts().to_dict() ## check after the transform


## prepare to calculate the percentage of orders using a coupon or gc for the same customer
dataset['Coupon_Code_Percent'] = np.where(dataset['Coupon_Code']=='none',0,1) 


## group by email address to see the number of orders placed by each of the customer, create another categorical variable called frequency
dataset["Purchase_Frequency"] = dataset.groupby(dataset['Email_Billing'])['Order_Number'].transform('count')

dataset.Purchase_Frequency.value_counts().to_dict()

## The maxiumn is 69 orders/year, the score indicate a relative frequency value compaire with the most frequent buyer
dataset['Purchase_Frequency_Score'] = round(dataset['Purchase_Frequency'] / 69 * 100,2) 


## Groupby email (which means customer_id) ##
dataset_customer = dataset.copy()

dataset_customer["SKU"] = dataset_customer.groupby(dataset_customer['Email_Billing'])['SKU'].transform(lambda x: '-'.join(x))

List3_join = ["Item_Name", "Order_Date", "Shipping_Method_Title", "Coupon_Code"]

List4_mean = ["Coupon_Code_Percent", "FreeShip_Percent"]

list5_sum = ["Order_Subtotal_Amount", "Cart_Discount_Amount","Order_Shipping_Amount", "Quantity"]

## Group the purchase items (and other variables) for one customer (note that a customer can place multiple orders in a year)
for x in List3_join:
    dataset_customer[x] = dataset_customer.groupby(dataset_customer['Email_Billing'])[x].transform(lambda x: ','.join(x))

## get the average using coupon rate and free-shipping rate for each customer (with multiple orders)
for x in List4_mean:
    dataset_customer[x] = 1 - round(dataset_customer.groupby(dataset_customer['Email_Billing'])[x].transform('mean'),2)
    
## Get the total order amount and total quantity for each customer
for x in list5_sum:
    dataset_customer[x] = dataset_customer.groupby(dataset_customer['Email_Billing'])[x].transform('sum')
    
## Drop duplicated orders, making one row stand for one customer
dataset_customer = dataset_customer.drop_duplicates(subset='Email_Billing', keep="first",ignore_index='True')

## check after grouping and aggregating
dataset_customer.head(5)


dataset_customer['Loyal_Customer'] = np.where(dataset_customer['Purchase_Frequency']==1,0,1)

print(dataset_customer.Loyal_Customer.value_counts())

dataset_customer['Discount_Rate'] = round(dataset_customer['Cart_Discount_Amount'] / dataset_customer['Order_Subtotal_Amount'] * 100, 2)

to_keep = ['Order_Subtotal_Amount','Discount_Rate','Order_Shipping_Amount','FreeShip_Percent','Coupon_Code_Percent','Age','Gender','Quantity','Loyal_Customer']

sub_data_1 = dataset_customer[to_keep]

sub_data_1.to_csv('output3.csv') ## for Classification

sub_data_2 = dataset_customer.loc[:, ['Item_Name']]

sub_data_2.to_csv('output1_c.csv') ## for Apirior
## Then use subline to remove all "" so the products can be seperate by ',' in to different columns




