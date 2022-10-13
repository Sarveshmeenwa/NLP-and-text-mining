```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns

from pandas.plotting import parallel_coordinates

```


```python
# !pip install mlxtend
#https://goldinlocks.github.io/Market-Basket-Analysis-in-Python/
```


```python
df = pd.read_csv("tel_samp_rec.csv",encoding="latin-1")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Defence.date</th>
      <th>Domains</th>
      <th>Full.Text.Language</th>
      <th>def.date</th>
      <th>n.disc</th>
      <th>these.id</th>
      <th>disc1.lev1</th>
      <th>disc1.lev2</th>
      <th>disc1.lev3</th>
      <th>disc2.lev1</th>
      <th>...</th>
      <th>n.tag</th>
      <th>disc1.rec.lev1</th>
      <th>disc1.rec.lev2</th>
      <th>disc1.rec.lev3</th>
      <th>disc2.rec.lev1</th>
      <th>disc2.rec.lev2</th>
      <th>disc2.rec.lev3</th>
      <th>disc3.rec.lev1</th>
      <th>disc3.rec.lev2</th>
      <th>disc3.rec.lev3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010/09/23</td>
      <td>Sciences du Vivant [q-bio] / Ecologie, Environ...</td>
      <td>French</td>
      <td>2010.0</td>
      <td>1</td>
      <td>tel-00662843v1</td>
      <td>Sciences du Vivant [q-bio]</td>
      <td>Ecologie, Environnement</td>
      <td>Ecosystèmes</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>X</td>
      <td>67 - Biologie des populations et écologie</td>
      <td>Ecologie, Environnement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009/11/02</td>
      <td>Sciences de l'Homme et Société</td>
      <td>French</td>
      <td>2009.0</td>
      <td>1</td>
      <td>tel-00491490v1</td>
      <td>Sciences de l'Homme et Société</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1</td>
      <td>IV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1996/05/30</td>
      <td>Sciences du Vivant [q-bio] / Alimentation et N...</td>
      <td>French</td>
      <td>1996.0</td>
      <td>1</td>
      <td>tel-01776364v1</td>
      <td>Sciences du Vivant [q-bio]</td>
      <td>Alimentation et Nutrition</td>
      <td>NaN</td>
      <td>Sciences du Vivant [q-bio]</td>
      <td>...</td>
      <td>2</td>
      <td>X</td>
      <td>68 - Biologie des organismes</td>
      <td>Alimentation et Nutrition</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018/02/02</td>
      <td>Informatique [cs] / Autre [cs.OH]  \r\n\r\nInf...</td>
      <td>French</td>
      <td>2018.0</td>
      <td>1</td>
      <td>tel-02437294v1</td>
      <td>Informatique [cs]</td>
      <td>Autre [cs.OH]</td>
      <td>NaN</td>
      <td>Informatique [cs]</td>
      <td>...</td>
      <td>2</td>
      <td>V</td>
      <td>27 - Informatique</td>
      <td>NaN</td>
      <td>V</td>
      <td>27 - Informatique</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015/07/08</td>
      <td>Informatique [cs] / Automatique  \r\n\r\nInfor...</td>
      <td>French</td>
      <td>2015.0</td>
      <td>1</td>
      <td>tel-01245100v1</td>
      <td>Informatique [cs]</td>
      <td>Automatique</td>
      <td>NaN</td>
      <td>Informatique [cs]</td>
      <td>...</td>
      <td>2</td>
      <td>V</td>
      <td>27 - Informatique</td>
      <td>NaN</td>
      <td>V</td>
      <td>27 - Informatique</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
cols = ['disc1.rec.lev1','disc2.rec.lev1','disc3.rec.lev1']
#subset columns shown above and take columns where all the 3 columns are not null
df_sub = df[df[cols].notnull().all(axis=1)]
```


```python
df_sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Defence.date</th>
      <th>Domains</th>
      <th>Full.Text.Language</th>
      <th>def.date</th>
      <th>n.disc</th>
      <th>these.id</th>
      <th>disc1.lev1</th>
      <th>disc1.lev2</th>
      <th>disc1.lev3</th>
      <th>disc2.lev1</th>
      <th>...</th>
      <th>n.tag</th>
      <th>disc1.rec.lev1</th>
      <th>disc1.rec.lev2</th>
      <th>disc1.rec.lev3</th>
      <th>disc2.rec.lev1</th>
      <th>disc2.rec.lev2</th>
      <th>disc2.rec.lev3</th>
      <th>disc3.rec.lev1</th>
      <th>disc3.rec.lev2</th>
      <th>disc3.rec.lev3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>1985/10/28</td>
      <td>Planète et Univers [physics] / Sciences de la ...</td>
      <td>French</td>
      <td>1985.0</td>
      <td>2</td>
      <td>tel-00711880v1</td>
      <td>Planète et Univers [physics]</td>
      <td>Sciences de la Terre</td>
      <td>Tectonique</td>
      <td>Sciences de l'environnement</td>
      <td>...</td>
      <td>3</td>
      <td>VIII</td>
      <td>35-36 - 2 sections - Sciences de la Terre</td>
      <td>Sciences de la Terre</td>
      <td>VIII</td>
      <td>37 - Météorologie, océanographie physique et p...</td>
      <td>Milieux et Changements globaux</td>
      <td>VIII</td>
      <td>35-36 - 2 sections - Sciences de la Terre</td>
      <td>Sciences de la Terre</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2018/12/17</td>
      <td>Sciences de l'ingénieur [physics] / Génie civi...</td>
      <td>English</td>
      <td>2018.0</td>
      <td>2</td>
      <td>tel-02182014v1</td>
      <td>Sciences de l'ingénieur [physics]</td>
      <td>Génie civil</td>
      <td>NaN</td>
      <td>Sciences de l'ingénieur [physics]</td>
      <td>...</td>
      <td>3</td>
      <td>IX</td>
      <td>60 - Mécanique, génie mécanique, génie civil</td>
      <td>Génie civil</td>
      <td>IX</td>
      <td>60 - Mécanique, génie mécanique, génie civil</td>
      <td>Génie civil</td>
      <td>V</td>
      <td>27 - Informatique</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>113</th>
      <td>2003/06/17</td>
      <td>Sciences de l'ingénieur [physics] / Traitement...</td>
      <td>French</td>
      <td>2003.0</td>
      <td>3</td>
      <td>tel-00130932v1</td>
      <td>Sciences de l'ingénieur [physics]</td>
      <td>Traitement du signal et de l'image [eess.SP]</td>
      <td>NaN</td>
      <td>Sciences du Vivant [q-bio]</td>
      <td>...</td>
      <td>3</td>
      <td>IX</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>VI</td>
      <td>28 - Milieux denses et matériaux</td>
      <td>Ingénierie biomédicale</td>
      <td>V</td>
      <td>27 - Informatique</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>193</th>
      <td>1997/10/24</td>
      <td>Planète et Univers [physics] / Sciences de la ...</td>
      <td>French</td>
      <td>1997.0</td>
      <td>2</td>
      <td>tel-00675418v1</td>
      <td>Planète et Univers [physics]</td>
      <td>Sciences de la Terre</td>
      <td>Géochimie</td>
      <td>Sciences de l'environnement</td>
      <td>...</td>
      <td>3</td>
      <td>VIII</td>
      <td>35-36 - 2 sections - Sciences de la Terre</td>
      <td>Sciences de la Terre</td>
      <td>VIII</td>
      <td>37 - Météorologie, océanographie physique et p...</td>
      <td>Milieux et Changements globaux</td>
      <td>VIII</td>
      <td>35-36 - 2 sections - Sciences de la Terre</td>
      <td>Sciences de la Terre</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2002/12/13</td>
      <td>Sciences du Vivant [q-bio] / Autre [q-bio.OT] ...</td>
      <td>French</td>
      <td>2002.0</td>
      <td>2</td>
      <td>tel-00008546v1</td>
      <td>Sciences du Vivant [q-bio]</td>
      <td>Autre [q-bio.OT]</td>
      <td>NaN</td>
      <td>Sciences de l'ingénieur [physics]</td>
      <td>...</td>
      <td>3</td>
      <td>X</td>
      <td>NaN</td>
      <td>Autre</td>
      <td>IX</td>
      <td>60 - Mécanique, génie mécanique, génie civil</td>
      <td>Mécanique</td>
      <td>IX</td>
      <td>NaN</td>
      <td>Autre</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
df_sub = df_sub[cols]
```


```python
df_sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>disc1.rec.lev1</th>
      <th>disc2.rec.lev1</th>
      <th>disc3.rec.lev1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>VIII</td>
      <td>VIII</td>
      <td>VIII</td>
    </tr>
    <tr>
      <th>104</th>
      <td>IX</td>
      <td>IX</td>
      <td>V</td>
    </tr>
    <tr>
      <th>113</th>
      <td>IX</td>
      <td>VI</td>
      <td>V</td>
    </tr>
    <tr>
      <th>193</th>
      <td>VIII</td>
      <td>VIII</td>
      <td>VIII</td>
    </tr>
    <tr>
      <th>212</th>
      <td>X</td>
      <td>IX</td>
      <td>IX</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting the list of transactions from the dataset
transactions = []
for i in range(0, len(df_sub)):
    transactions.append([str(df_sub.values[i,j]) for j in range(0, len(df_sub.columns))])
```


```python
#check transactions
transactions[:1]

```




    [['VIII', 'VIII', 'VIII']]




```python

# Extract unique items.
flattened = [item for transaction in transactions for item in transaction]
items = list(set(flattened))
```


```python
print('# of items:',len(items))
print(list(items))
```

    # of items: 13
    ['IV', 'pharmacie', 'VI', 'I', 'III', 'XII', 'II', 'VIII', 'X', 'V', 'I - Droit', 'VII', 'IX']
    


```python
#remove nan if present in list
if 'nan' in items: items.remove('nan')
print(list(items))
```

    ['IV', 'pharmacie', 'VI', 'I', 'III', 'XII', 'II', 'VIII', 'X', 'V', 'I - Droit', 'VII', 'IX']
    


```python
# Compute and print rules.
rules = list(permutations(items, 2))
print('# of rules:',len(rules))
print(rules[:5])
```

    # of rules: 156
    [('IV', 'pharmacie'), ('IV', 'VI'), ('IV', 'I'), ('IV', 'III'), ('IV', 'XII')]
    


```python
# Import the transaction encoder function from mlxtend
from mlxtend.preprocessing import TransactionEncoder

# Instantiate transaction encoder and identify unique items
encoder = TransactionEncoder().fit(transactions)

# One-hot encode transactions
onehot = encoder.transform(transactions)

# Convert one-hot encoded data to DataFrame
onehot = pd.DataFrame(onehot, columns = encoder.columns_)

# Print the one-hot encoded transaction dataset
onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>I</th>
      <th>I - Droit</th>
      <th>II</th>
      <th>III</th>
      <th>IV</th>
      <th>IX</th>
      <th>V</th>
      <th>VI</th>
      <th>VII</th>
      <th>VIII</th>
      <th>X</th>
      <th>XII</th>
      <th>pharmacie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
def leverage(antecedent, consequent):
    # Compute support for antecedent AND consequent
    supportAB = np.logical_and(antecedent, consequent).mean()

    # Compute support for antecedent
    supportA = antecedent.mean()

    # Compute support for consequent
    supportB = consequent.mean()

    # Return leverage
    return supportAB - supportB * supportA

# Define a function to compute Zhang's metric
def zhang(antecedent, consequent):
    # Compute the support of each book
    supportA = antecedent.mean()
    supportC = consequent.mean()

    # Compute the support of both books
    supportAC = np.logical_and(antecedent, consequent).mean()

    # Complete the expressions for the numerator and denominator
    numerator = supportAC - supportA*supportC
    denominator = max(supportAC*(1-supportA), supportA*(supportC-supportAC))

    # Return Zhang's metric
    return numerator / denominator

def conviction(antecedent, consequent):
    # Compute support for antecedent AND consequent
    supportAC = np.logical_and(antecedent, consequent).mean()

    # Compute support for antecedent
    supportA = antecedent.mean()

    # Compute support for NOT consequent
    supportnC = 1.0 - consequent.mean()

    # Compute support for antecedent and NOT consequent
    supportAnC = supportA - supportAC

    # Return conviction
    return supportA * supportnC / supportAnC

```


```python
# Create rules DataFrame
rules_ = pd.DataFrame(rules, columns=['antecedents','consequents'])

# Define an empty list for metrics
zhangs, conv, lev, antec_supp, cons_supp, suppt, conf, lft = [], [], [], [], [], [], [], []

# Loop over lists in itemsets
for itemset in rules:
    # Extract the antecedent and consequent columns
    antecedent = onehot[itemset[0]]
    consequent = onehot[itemset[1]]
    
    antecedent_support = onehot[itemset[0]].mean()
    consequent_support = onehot[itemset[1]].mean()
    support = np.logical_and(onehot[itemset[0]], onehot[itemset[1]]).mean()
    confidence = support / antecedent_support
    lift = support / (antecedent_support * consequent_support)
    
    # Complete metrics and append it to the list
    antec_supp.append(antecedent_support)
    cons_supp.append(consequent_support)
    suppt.append(support)
    conf.append(confidence)
    lft.append(lift)
    lev.append(leverage(antecedent, consequent))
    conv.append(conviction(antecedent, consequent))
    zhangs.append(zhang(antecedent, consequent))
    
# Store results
rules_['antecedent support'] = antec_supp
rules_['consequent support'] = cons_supp
rules_['support'] = suppt
rules_['confidence'] = conf
rules_['lift'] = lft
rules_['leverage'] = lev
rules_['conviction'] = conv
rules_['zhang'] = zhangs

# Print results
rules_.sort_values('zhang',ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>IV</td>
      <td>I</td>
      <td>0.143959</td>
      <td>0.021994</td>
      <td>0.020566</td>
      <td>0.142857</td>
      <td>6.495362</td>
      <td>0.017399</td>
      <td>1.141007</td>
      <td>0.988322</td>
    </tr>
    <tr>
      <th>9</th>
      <td>IV</td>
      <td>I - Droit</td>
      <td>0.143959</td>
      <td>0.004856</td>
      <td>0.003999</td>
      <td>0.027778</td>
      <td>5.720588</td>
      <td>0.003300</td>
      <td>1.023577</td>
      <td>0.963964</td>
    </tr>
    <tr>
      <th>97</th>
      <td>X</td>
      <td>pharmacie</td>
      <td>0.199372</td>
      <td>0.019709</td>
      <td>0.016281</td>
      <td>0.081662</td>
      <td>4.143453</td>
      <td>0.012352</td>
      <td>1.067462</td>
      <td>0.947575</td>
    </tr>
    <tr>
      <th>45</th>
      <td>I</td>
      <td>I - Droit</td>
      <td>0.021994</td>
      <td>0.004856</td>
      <td>0.001428</td>
      <td>0.064935</td>
      <td>13.372804</td>
      <td>0.001321</td>
      <td>1.064251</td>
      <td>0.946028</td>
    </tr>
    <tr>
      <th>123</th>
      <td>I - Droit</td>
      <td>I</td>
      <td>0.004856</td>
      <td>0.021994</td>
      <td>0.001428</td>
      <td>0.294118</td>
      <td>13.372804</td>
      <td>0.001321</td>
      <td>1.385509</td>
      <td>0.929736</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Function to convert rules to coordinates.
def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]
```


```python

```


```python
rules_.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>zhang</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IV</td>
      <td>pharmacie</td>
      <td>0.143959</td>
      <td>0.019709</td>
      <td>0.000857</td>
      <td>0.005952</td>
      <td>0.302019</td>
      <td>-0.001980</td>
      <td>0.986161</td>
      <td>-0.729708</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IV</td>
      <td>VI</td>
      <td>0.143959</td>
      <td>0.162239</td>
      <td>0.000857</td>
      <td>0.005952</td>
      <td>0.036689</td>
      <td>-0.022499</td>
      <td>0.842777</td>
      <td>-0.968426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IV</td>
      <td>I</td>
      <td>0.143959</td>
      <td>0.021994</td>
      <td>0.020566</td>
      <td>0.142857</td>
      <td>6.495362</td>
      <td>0.017399</td>
      <td>1.141007</td>
      <td>0.988322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IV</td>
      <td>III</td>
      <td>0.143959</td>
      <td>0.032562</td>
      <td>0.022279</td>
      <td>0.154762</td>
      <td>4.752820</td>
      <td>0.017592</td>
      <td>1.144574</td>
      <td>0.922384</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IV</td>
      <td>XII</td>
      <td>0.143959</td>
      <td>0.049414</td>
      <td>0.034276</td>
      <td>0.238095</td>
      <td>4.818332</td>
      <td>0.027162</td>
      <td>1.247644</td>
      <td>0.925726</td>
    </tr>
  </tbody>
</table>
</div>




```python
#remove rows where antecedent = consequent
rules_ = rules_[rules_['antecedents'] != rules_['consequents']]
```


```python
#filter rules with lift > 1
rules_ = rules_.query("lift>1")
#create support table based on lift values > 1
support_table = rules_.pivot(index='consequents', columns='antecedents',
values='lift')




sns.heatmap(support_table)
```




    <AxesSubplot:xlabel='antecedents', ylabel='consequents'>




    
![png](output_21_1.png)
    



```python
# Generate frequent itemsets
frequent_itemsets = apriori(onehot, min_support = 0.01, use_colnames = True, max_len = 2)
# Generate association rules
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1.00)
# Generate coordinates and print example
coords = rules_to_coordinates(rules)
# Generate parallel coordinates plot

plt.figure(figsize=(4,8))
parallel_coordinates(coords, 'rule')
plt.legend([])
plt.grid(True)
plt.show()
```


    
![png](output_22_0.png)
    


 https://www.galaxie.enseignementsup-recherche.gouv.fr/ensup/pdf/qualification/sections.pdf
> LINK : Reference what I , II etc means



```python

```
