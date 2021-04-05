```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
sns.set(style='ticks', color_codes = True)
```


```python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
```

Importing DataSet


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

Appending train and test for ease of data cleaning


```python
df = train.append(test , sort = True)
```

some data exploration


```python
print('Train:',train.shape)
print('Test:',test.shape)
print('DataFrame:',df.shape)
```

    Train: (1460, 81)
    Test: (1459, 80)
    DataFrame: (2919, 81)
    


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
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>Alley</th>
      <th>BedroomAbvGr</th>
      <th>BldgType</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtQual</th>
      <th>BsmtUnfSF</th>
      <th>CentralAir</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>Electrical</th>
      <th>EnclosedPorch</th>
      <th>ExterCond</th>
      <th>ExterQual</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>Fence</th>
      <th>FireplaceQu</th>
      <th>Fireplaces</th>
      <th>Foundation</th>
      <th>FullBath</th>
      <th>Functional</th>
      <th>GarageArea</th>
      <th>GarageCars</th>
      <th>GarageCond</th>
      <th>GarageFinish</th>
      <th>GarageQual</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GrLivArea</th>
      <th>HalfBath</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>HouseStyle</th>
      <th>Id</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>LandContour</th>
      <th>LandSlope</th>
      <th>LotArea</th>
      <th>LotConfig</th>
      <th>LotFrontage</th>
      <th>LotShape</th>
      <th>LowQualFinSF</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>MasVnrArea</th>
      <th>MasVnrType</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>Neighborhood</th>
      <th>OpenPorchSF</th>
      <th>OverallCond</th>
      <th>OverallQual</th>
      <th>PavedDrive</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>RoofMatl</th>
      <th>RoofStyle</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
      <th>SaleType</th>
      <th>ScreenPorch</th>
      <th>Street</th>
      <th>TotRmsAbvGrd</th>
      <th>TotalBsmtSF</th>
      <th>Utilities</th>
      <th>WoodDeckSF</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>No</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>150.0</td>
      <td>Y</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>PConc</td>
      <td>2</td>
      <td>Typ</td>
      <td>548.0</td>
      <td>2.0</td>
      <td>TA</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>1710</td>
      <td>1</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>2Story</td>
      <td>1</td>
      <td>1</td>
      <td>Gd</td>
      <td>Lvl</td>
      <td>Gtl</td>
      <td>8450</td>
      <td>Inside</td>
      <td>65.0</td>
      <td>Reg</td>
      <td>0</td>
      <td>60</td>
      <td>RL</td>
      <td>196.0</td>
      <td>BrkFace</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>CollgCr</td>
      <td>61</td>
      <td>5</td>
      <td>7</td>
      <td>Y</td>
      <td>0</td>
      <td>NaN</td>
      <td>CompShg</td>
      <td>Gable</td>
      <td>Normal</td>
      <td>208500.0</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>8</td>
      <td>856.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2003</td>
      <td>2003</td>
      <td>2008</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Gd</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>ALQ</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Gd</td>
      <td>284.0</td>
      <td>Y</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>TA</td>
      <td>TA</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>NaN</td>
      <td>TA</td>
      <td>1</td>
      <td>CBlock</td>
      <td>2</td>
      <td>Typ</td>
      <td>460.0</td>
      <td>2.0</td>
      <td>TA</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>1262</td>
      <td>0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>1Story</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>Lvl</td>
      <td>Gtl</td>
      <td>9600</td>
      <td>FR2</td>
      <td>80.0</td>
      <td>Reg</td>
      <td>0</td>
      <td>20</td>
      <td>RL</td>
      <td>0.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>Veenker</td>
      <td>0</td>
      <td>8</td>
      <td>6</td>
      <td>Y</td>
      <td>0</td>
      <td>NaN</td>
      <td>CompShg</td>
      <td>Gable</td>
      <td>Normal</td>
      <td>181500.0</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>1262.0</td>
      <td>AllPub</td>
      <td>298</td>
      <td>1976</td>
      <td>1976</td>
      <td>2007</td>
    </tr>
    <tr>
      <td>2</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Mn</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>434.0</td>
      <td>Y</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>TA</td>
      <td>1</td>
      <td>PConc</td>
      <td>2</td>
      <td>Typ</td>
      <td>608.0</td>
      <td>2.0</td>
      <td>TA</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>1786</td>
      <td>1</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>2Story</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>Lvl</td>
      <td>Gtl</td>
      <td>11250</td>
      <td>Inside</td>
      <td>68.0</td>
      <td>IR1</td>
      <td>0</td>
      <td>60</td>
      <td>RL</td>
      <td>162.0</td>
      <td>BrkFace</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>CollgCr</td>
      <td>42</td>
      <td>5</td>
      <td>7</td>
      <td>Y</td>
      <td>0</td>
      <td>NaN</td>
      <td>CompShg</td>
      <td>Gable</td>
      <td>Normal</td>
      <td>223500.0</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>920.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2001</td>
      <td>2002</td>
      <td>2008</td>
    </tr>
    <tr>
      <td>3</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>Gd</td>
      <td>No</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>ALQ</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>TA</td>
      <td>540.0</td>
      <td>Y</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>SBrkr</td>
      <td>272</td>
      <td>TA</td>
      <td>TA</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>NaN</td>
      <td>Gd</td>
      <td>1</td>
      <td>BrkTil</td>
      <td>1</td>
      <td>Typ</td>
      <td>642.0</td>
      <td>3.0</td>
      <td>TA</td>
      <td>Unf</td>
      <td>TA</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>1717</td>
      <td>0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>2Story</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>Lvl</td>
      <td>Gtl</td>
      <td>9550</td>
      <td>Corner</td>
      <td>60.0</td>
      <td>IR1</td>
      <td>0</td>
      <td>70</td>
      <td>RL</td>
      <td>0.0</td>
      <td>None</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>Crawfor</td>
      <td>35</td>
      <td>5</td>
      <td>7</td>
      <td>Y</td>
      <td>0</td>
      <td>NaN</td>
      <td>CompShg</td>
      <td>Gable</td>
      <td>Abnorml</td>
      <td>140000.0</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>7</td>
      <td>756.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>1915</td>
      <td>1970</td>
      <td>2006</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Av</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>GLQ</td>
      <td>Unf</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>490.0</td>
      <td>Y</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>SBrkr</td>
      <td>0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>TA</td>
      <td>1</td>
      <td>PConc</td>
      <td>2</td>
      <td>Typ</td>
      <td>836.0</td>
      <td>3.0</td>
      <td>TA</td>
      <td>RFn</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>2198</td>
      <td>1</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>2Story</td>
      <td>5</td>
      <td>1</td>
      <td>Gd</td>
      <td>Lvl</td>
      <td>Gtl</td>
      <td>14260</td>
      <td>FR2</td>
      <td>84.0</td>
      <td>IR1</td>
      <td>0</td>
      <td>60</td>
      <td>RL</td>
      <td>350.0</td>
      <td>BrkFace</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>NoRidge</td>
      <td>84</td>
      <td>5</td>
      <td>8</td>
      <td>Y</td>
      <td>0</td>
      <td>NaN</td>
      <td>CompShg</td>
      <td>Gable</td>
      <td>Normal</td>
      <td>250000.0</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>9</td>
      <td>1145.0</td>
      <td>AllPub</td>
      <td>192</td>
      <td>2000</td>
      <td>2000</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    1stFlrSF            0
    2ndFlrSF            0
    3SsnPorch           0
    Alley            2721
    BedroomAbvGr        0
    BldgType            0
    BsmtCond           82
    BsmtExposure       82
    BsmtFinSF1          1
    BsmtFinSF2          1
    BsmtFinType1       79
    BsmtFinType2       80
    BsmtFullBath        2
    BsmtHalfBath        2
    BsmtQual           81
    BsmtUnfSF           1
    CentralAir          0
    Condition1          0
    Condition2          0
    Electrical          1
    EnclosedPorch       0
    ExterCond           0
    ExterQual           0
    Exterior1st         1
    Exterior2nd         1
    Fence            2348
    FireplaceQu      1420
    Fireplaces          0
    Foundation          0
    FullBath            0
    Functional          2
    GarageArea          1
    GarageCars          1
    GarageCond        159
    GarageFinish      159
    GarageQual        159
    GarageType        157
    GarageYrBlt       159
    GrLivArea           0
    HalfBath            0
    Heating             0
    HeatingQC           0
    HouseStyle          0
    Id                  0
    KitchenAbvGr        0
    KitchenQual         1
    LandContour         0
    LandSlope           0
    LotArea             0
    LotConfig           0
    LotFrontage       486
    LotShape            0
    LowQualFinSF        0
    MSSubClass          0
    MSZoning            4
    MasVnrArea         23
    MasVnrType         24
    MiscFeature      2814
    MiscVal             0
    MoSold              0
    Neighborhood        0
    OpenPorchSF         0
    OverallCond         0
    OverallQual         0
    PavedDrive          0
    PoolArea            0
    PoolQC           2909
    RoofMatl            0
    RoofStyle           0
    SaleCondition       0
    SalePrice        1459
    SaleType            1
    ScreenPorch         0
    Street              0
    TotRmsAbvGrd        0
    TotalBsmtSF         1
    Utilities           2
    WoodDeckSF          0
    YearBuilt           0
    YearRemodAdd        0
    YrSold              0
    dtype: int64



there are a lot of missing values, let's seperate columns into numerical and categorical


```python
df.dtypes
```




    1stFlrSF           int64
    2ndFlrSF           int64
    3SsnPorch          int64
    Alley             object
    BedroomAbvGr       int64
    BldgType          object
    BsmtCond          object
    BsmtExposure      object
    BsmtFinSF1       float64
    BsmtFinSF2       float64
    BsmtFinType1      object
    BsmtFinType2      object
    BsmtFullBath     float64
    BsmtHalfBath     float64
    BsmtQual          object
    BsmtUnfSF        float64
    CentralAir        object
    Condition1        object
    Condition2        object
    Electrical        object
    EnclosedPorch      int64
    ExterCond         object
    ExterQual         object
    Exterior1st       object
    Exterior2nd       object
    Fence             object
    FireplaceQu       object
    Fireplaces         int64
    Foundation        object
    FullBath           int64
    Functional        object
    GarageArea       float64
    GarageCars       float64
    GarageCond        object
    GarageFinish      object
    GarageQual        object
    GarageType        object
    GarageYrBlt      float64
    GrLivArea          int64
    HalfBath           int64
    Heating           object
    HeatingQC         object
    HouseStyle        object
    Id                 int64
    KitchenAbvGr       int64
    KitchenQual       object
    LandContour       object
    LandSlope         object
    LotArea            int64
    LotConfig         object
    LotFrontage      float64
    LotShape          object
    LowQualFinSF       int64
    MSSubClass         int64
    MSZoning          object
    MasVnrArea       float64
    MasVnrType        object
    MiscFeature       object
    MiscVal            int64
    MoSold             int64
    Neighborhood      object
    OpenPorchSF        int64
    OverallCond        int64
    OverallQual        int64
    PavedDrive        object
    PoolArea           int64
    PoolQC            object
    RoofMatl          object
    RoofStyle         object
    SaleCondition     object
    SalePrice        float64
    SaleType          object
    ScreenPorch        int64
    Street            object
    TotRmsAbvGrd       int64
    TotalBsmtSF      float64
    Utilities         object
    WoodDeckSF         int64
    YearBuilt          int64
    YearRemodAdd       int64
    YrSold             int64
    dtype: object




```python
cat = list(df.select_dtypes('object'))
num = list(df.select_dtypes(['int64','float64']))
```


```python
cat
```




    ['Alley',
     'BldgType',
     'BsmtCond',
     'BsmtExposure',
     'BsmtFinType1',
     'BsmtFinType2',
     'BsmtQual',
     'CentralAir',
     'Condition1',
     'Condition2',
     'Electrical',
     'ExterCond',
     'ExterQual',
     'Exterior1st',
     'Exterior2nd',
     'Fence',
     'FireplaceQu',
     'Foundation',
     'Functional',
     'GarageCond',
     'GarageFinish',
     'GarageQual',
     'GarageType',
     'Heating',
     'HeatingQC',
     'HouseStyle',
     'KitchenQual',
     'LandContour',
     'LandSlope',
     'LotConfig',
     'LotShape',
     'MSZoning',
     'MasVnrType',
     'MiscFeature',
     'Neighborhood',
     'PavedDrive',
     'PoolQC',
     'RoofMatl',
     'RoofStyle',
     'SaleCondition',
     'SaleType',
     'Street',
     'Utilities']




```python
num
```




    ['1stFlrSF',
     '2ndFlrSF',
     '3SsnPorch',
     'BedroomAbvGr',
     'BsmtFinSF1',
     'BsmtFinSF2',
     'BsmtFullBath',
     'BsmtHalfBath',
     'BsmtUnfSF',
     'EnclosedPorch',
     'Fireplaces',
     'FullBath',
     'GarageArea',
     'GarageCars',
     'GarageYrBlt',
     'GrLivArea',
     'HalfBath',
     'Id',
     'KitchenAbvGr',
     'LotArea',
     'LotFrontage',
     'LowQualFinSF',
     'MSSubClass',
     'MasVnrArea',
     'MiscVal',
     'MoSold',
     'OpenPorchSF',
     'OverallCond',
     'OverallQual',
     'PoolArea',
     'SalePrice',
     'ScreenPorch',
     'TotRmsAbvGrd',
     'TotalBsmtSF',
     'WoodDeckSF',
     'YearBuilt',
     'YearRemodAdd',
     'YrSold']




```python
na = df[num].isnull().sum()
na = na[na > 0]
na = na.sort_values(ascending=False)
print(na)
```

    SalePrice       1459
    LotFrontage      486
    GarageYrBlt      159
    MasVnrArea        23
    BsmtHalfBath       2
    BsmtFullBath       2
    TotalBsmtSF        1
    GarageCars         1
    GarageArea         1
    BsmtUnfSF          1
    BsmtFinSF2         1
    BsmtFinSF1         1
    dtype: int64
    


```python
na = df[cat].isnull().sum()
na = na[na > 0]
na = na.sort_values(ascending=False)
print(na)
```

    PoolQC          2909
    MiscFeature     2814
    Alley           2721
    Fence           2348
    FireplaceQu     1420
    GarageQual       159
    GarageFinish     159
    GarageCond       159
    GarageType       157
    BsmtCond          82
    BsmtExposure      82
    BsmtQual          81
    BsmtFinType2      80
    BsmtFinType1      79
    MasVnrType        24
    MSZoning           4
    Utilities          2
    Functional         2
    Electrical         1
    Exterior1st        1
    Exterior2nd        1
    SaleType           1
    KitchenQual        1
    dtype: int64
    

Numerical columns missing values


```python
df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)
df.GarageYrBlt.fillna(0, inplace=True)
```

above we replaced missing values in LotFrontage with median so that donot change shape of data

in GarageYrBlt null value means there is no garage so we replaced it with 0.

in all other numerical columns null value indicate not available so we replace it with 0 below.


```python
df.MasVnrArea.fillna(0, inplace=True)    
df.BsmtHalfBath.fillna(0, inplace=True)
df.BsmtFullBath.fillna(0, inplace=True)
df.GarageArea.fillna(0, inplace=True)
df.GarageCars.fillna(0, inplace=True)    
df.TotalBsmtSF.fillna(0, inplace=True)   
df.BsmtUnfSF.fillna(0, inplace=True)     
df.BsmtFinSF2.fillna(0, inplace=True)    
df.BsmtFinSF1.fillna(0, inplace=True) 
```


```python
df[num].isnull().sum()
```




    1stFlrSF            0
    2ndFlrSF            0
    3SsnPorch           0
    BedroomAbvGr        0
    BsmtFinSF1          0
    BsmtFinSF2          0
    BsmtFullBath        0
    BsmtHalfBath        0
    BsmtUnfSF           0
    EnclosedPorch       0
    Fireplaces          0
    FullBath            0
    GarageArea          0
    GarageCars          0
    GarageYrBlt         0
    GrLivArea           0
    HalfBath            0
    Id                  0
    KitchenAbvGr        0
    LotArea             0
    LotFrontage         0
    LowQualFinSF        0
    MSSubClass          0
    MasVnrArea          0
    MiscVal             0
    MoSold              0
    OpenPorchSF         0
    OverallCond         0
    OverallQual         0
    PoolArea            0
    SalePrice        1459
    ScreenPorch         0
    TotRmsAbvGrd        0
    TotalBsmtSF         0
    WoodDeckSF          0
    YearBuilt           0
    YearRemodAdd        0
    YrSold              0
    dtype: int64



so all missing values treated exept SalePrice, Here SalePrice is that of test data which we have to predict.

Categorical columns missing values


```python
df.PoolQC.fillna('NA', inplace=True)
df.MiscFeature.fillna('NA', inplace=True)    
df.Alley.fillna('NA', inplace=True)          
df.Fence.fillna('NA', inplace=True)         
df.FireplaceQu.fillna('NA', inplace=True)    
df.GarageCond.fillna('NA', inplace=True)    
df.GarageQual.fillna('NA', inplace=True)     
df.GarageFinish.fillna('NA', inplace=True)   
df.GarageType.fillna('NA', inplace=True)     
df.BsmtExposure.fillna('NA', inplace=True)     
df.BsmtCond.fillna('NA', inplace=True)        
df.BsmtQual.fillna('NA', inplace=True)        
df.BsmtFinType2.fillna('NA', inplace=True)     
df.BsmtFinType1.fillna('NA', inplace=True)     
df.MasVnrType.fillna('None', inplace=True)   
df.Exterior2nd.fillna('None', inplace=True) 
```

all these columns similar to numerical column where null value meant not available we replace it with NA or None according to the data dictionary and rest we fill the mode.


```python
df.Functional.fillna(df.Functional.mode()[0], inplace=True)       
df.Utilities.fillna(df.Utilities.mode()[0], inplace=True)          
df.Exterior1st.fillna(df.Exterior1st.mode()[0], inplace=True)        
df.SaleType.fillna(df.SaleType.mode()[0], inplace=True)                
df.KitchenQual.fillna(df.KitchenQual.mode()[0], inplace=True)        
df.Electrical.fillna(df.Electrical.mode()[0], inplace=True) 
df.MSZoning.fillna(df.MSZoning.mode()[0], inplace=True)
```


```python
df[cat].isnull().sum()
```




    Alley            0
    BldgType         0
    BsmtCond         0
    BsmtExposure     0
    BsmtFinType1     0
    BsmtFinType2     0
    BsmtQual         0
    CentralAir       0
    Condition1       0
    Condition2       0
    Electrical       0
    ExterCond        0
    ExterQual        0
    Exterior1st      0
    Exterior2nd      0
    Fence            0
    FireplaceQu      0
    Foundation       0
    Functional       0
    GarageCond       0
    GarageFinish     0
    GarageQual       0
    GarageType       0
    Heating          0
    HeatingQC        0
    HouseStyle       0
    KitchenQual      0
    LandContour      0
    LandSlope        0
    LotConfig        0
    LotShape         0
    MSZoning         0
    MasVnrType       0
    MiscFeature      0
    Neighborhood     0
    PavedDrive       0
    PoolQC           0
    RoofMatl         0
    RoofStyle        0
    SaleCondition    0
    SaleType         0
    Street           0
    Utilities        0
    dtype: int64



all null values treated with no missing variable currently.


```python
def boxplot(var):
    sns.catplot(x=var, y='SalePrice',data = train, kind='box')
```


```python
boxplot('Alley')
boxplot('BldgType')
boxplot('BsmtCond')
boxplot('BsmtFinType1')
```


![png](output_30_0.png)



![png](output_30_1.png)



![png](output_30_2.png)



![png](output_30_3.png)


Alley has a impact on price as we can see from boxplot. 
BldgType,BsmtCond,BsmtFinType1 has a very small impact.


```python
boxplot('BsmtExposure')
boxplot('BsmtFinType2')
boxplot('BsmtQual')
boxplot('CentralAir')
```


![png](output_32_0.png)



![png](output_32_1.png)



![png](output_32_2.png)



![png](output_32_3.png)


CentralAir,BsmtQual has a impact on price as we can see from boxplot. BsmtFinType2,BsmtExposure has a very small impact.


```python
boxplot('Condition1')
boxplot('Condition2')
boxplot('Electrical')
boxplot('ExterCond')
```


![png](output_34_0.png)



![png](output_34_1.png)



![png](output_34_2.png)



![png](output_34_3.png)


ExterCond,Condition2,Condition1 has a impact on price as we can see from boxplot. Electrical has a very small impact.


```python
boxplot('ExterQual')
boxplot('Exterior1st')
boxplot('Exterior2nd')
boxplot('Fence')
```


![png](output_36_0.png)



![png](output_36_1.png)



![png](output_36_2.png)



![png](output_36_3.png)


ExterQual has a impact on price as we can see from boxplot. Fence,Exterior1st,Exterior2nd has a very small impact.


```python
boxplot('FireplaceQu')
boxplot('Foundation')
boxplot('Functional')
boxplot('GarageCond')
```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)



![png](output_38_3.png)


FireplaceQu,Foundation,Functional,GarageCond has a very small impact. also excellent fireplacequal shows a jump in price.


```python
boxplot('GarageFinish')
boxplot('GarageQual')
boxplot('GarageType')
boxplot('Heating')
```


![png](output_40_0.png)



![png](output_40_1.png)



![png](output_40_2.png)



![png](output_40_3.png)


GarageQual,GarageFinish has a impact on price as we can see from boxplot. Heating,GarageType has a very small impact.


```python
boxplot('HeatingQC')
boxplot('HouseStyle')
boxplot('KitchenQual')
boxplot('LandContour')
```


![png](output_42_0.png)



![png](output_42_1.png)



![png](output_42_2.png)



![png](output_42_3.png)


HouseStyle,KitchenQual has a impact on price as we can see from boxplot. LandContour,HeatingQC has a very small impact.


```python
boxplot('LandSlope')
boxplot('LotConfig')
boxplot('LotShape')
boxplot('MSZoning')
```


![png](output_44_0.png)



![png](output_44_1.png)



![png](output_44_2.png)



![png](output_44_3.png)


MSZoning has a impact on price as we can see from boxplot. LotShape,LotConfig,LandSlope has no impact.


```python
boxplot('MasVnrType')
boxplot('MiscFeature')
boxplot('Neighborhood')
boxplot('PavedDrive')
```


![png](output_46_0.png)



![png](output_46_1.png)



![png](output_46_2.png)



![png](output_46_3.png)


PavedDrive,Neighborhood,MiscFeature has a impact on price. MasVnrType has a very small impact,changes drastically with stone.


```python
boxplot('PoolQC')
boxplot('RoofMatl')
boxplot('RoofStyle')
boxplot('SaleCondition')
```


![png](output_48_0.png)



![png](output_48_1.png)



![png](output_48_2.png)



![png](output_48_3.png)


RoofStyle,RoofMatl,PoolQC has a impact on price as we can see from boxplot. SaleCondition has a very small impact and increase with Partial.


```python
boxplot('SaleType')
boxplot('Street')
boxplot('Utilities')
```


![png](output_50_0.png)



![png](output_50_1.png)



![png](output_50_2.png)


SaleType has a impact on price as we can see from boxplot. Street,Utilities has a very small impact.

#### High Dependency:
CentralAir  
BsmtQual  
Alley  
ExterCond  
Condition2  
Condition1  
ExterQual  
GarageQual  
GarageFinish  
HouseStyle  
KitchenQual  
MSZoning  
PavedDrive  
Neighborhood  
MiscFeature  
RoofStyle  
RoofMatl  
PoolQC  
SaleType  

#### Modarate Dependency:
BldgType 
BsmtCond  
BsmtFinType1  
BsmtFinType2  
BsmtExposure  
Electrical  
Fence  
Exterior1st  
Exterior2nd  
FireplaceQu  
Foundation  
Functional  
GarageCond  
Heating  
GarageType  
LandContour  
HeatingQC  
SaleCondition   
Street  
Utilities  

#### Low Dependency:
LotShape  
LotConfig  
LandSlope  
MasVnrType  


```python
corrmat = df.corr()
f, ax = plt.subplots(figsize=(24, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c89ad25c08>




![png](output_53_1.png)



```python
k = 20
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
f, ax = plt.subplots(figsize=(24, 9))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.1f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```


![png](output_54_0.png)


#### High Correlated
OverallQual  
GrLivArea  
GarageCars  
TotalBsmtSF  
1stFlrSF  
FullBath  
TotRmsAbvGrd  
YearBuilt  
YrRemodAdd  
MasVnrArea   
Fireplaces  
BsmtFinSF1  
LotFrontage  
WoodDeckSF  
2ndFlrSF  
OpenPorchSF  
HalfBath  
LotArea  
        

It's needed to create dummy vars and map categorical features in order to run ML model.

Create mapping for categorical features that can be ranked.


```python
df.Alley = df.Alley.map({'NA':0, 'Grvl':1, 'Pave':2})
df.BsmtCond =  df.BsmtCond.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.BsmtExposure = df.BsmtExposure.map({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})
df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
df.BsmtQual = df.BsmtQual.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.ExterCond = df.ExterCond.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.ExterQual = df.ExterQual.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.FireplaceQu = df.FireplaceQu.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.Functional = df.Functional.map({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8})
df.GarageCond = df.GarageCond.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.GarageQual = df.GarageQual.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.HeatingQC = df.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.KitchenQual = df.KitchenQual.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.LandSlope = df.LandSlope.map({'Sev':1, 'Mod':2, 'Gtl':3}) 
df.PavedDrive = df.PavedDrive.map({'N':1, 'P':2, 'Y':3})
df.PoolQC = df.PoolQC.map({'NA':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
df.Street = df.Street.map({'Grvl':1, 'Pave':2})
df.Utilities = df.Utilities.map({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4})
```


```python
new_num = ['Alley','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual',
           'ExterCond','ExterQual','FireplaceQu','Functional','GarageCond',
           'GarageQual','HeatingQC','KitchenQual','LandSlope','PavedDrive','PoolQC',
           'Street','Utilities']
num = num + new_num
for i in new_num:
    cat.remove(i)
```

all these are now numerical so added to num list and removed from cat list.


```python
num
```




    ['1stFlrSF',
     '2ndFlrSF',
     '3SsnPorch',
     'BedroomAbvGr',
     'BsmtFinSF1',
     'BsmtFinSF2',
     'BsmtFullBath',
     'BsmtHalfBath',
     'BsmtUnfSF',
     'EnclosedPorch',
     'Fireplaces',
     'FullBath',
     'GarageArea',
     'GarageCars',
     'GarageYrBlt',
     'GrLivArea',
     'HalfBath',
     'Id',
     'KitchenAbvGr',
     'LotArea',
     'LotFrontage',
     'LowQualFinSF',
     'MSSubClass',
     'MasVnrArea',
     'MiscVal',
     'MoSold',
     'OpenPorchSF',
     'OverallCond',
     'OverallQual',
     'PoolArea',
     'SalePrice',
     'ScreenPorch',
     'TotRmsAbvGrd',
     'TotalBsmtSF',
     'WoodDeckSF',
     'YearBuilt',
     'YearRemodAdd',
     'YrSold',
     'Alley',
     'BsmtCond',
     'BsmtExposure',
     'BsmtFinType1',
     'BsmtFinType2',
     'BsmtQual',
     'ExterCond',
     'ExterQual',
     'FireplaceQu',
     'Functional',
     'GarageCond',
     'GarageQual',
     'HeatingQC',
     'KitchenQual',
     'LandSlope',
     'PavedDrive',
     'PoolQC',
     'Street',
     'Utilities']




```python
cat
```




    ['BldgType',
     'CentralAir',
     'Condition1',
     'Condition2',
     'Electrical',
     'Exterior1st',
     'Exterior2nd',
     'Fence',
     'Foundation',
     'GarageFinish',
     'GarageType',
     'Heating',
     'HouseStyle',
     'LandContour',
     'LotConfig',
     'LotShape',
     'MSZoning',
     'MasVnrType',
     'MiscFeature',
     'Neighborhood',
     'RoofMatl',
     'RoofStyle',
     'SaleCondition',
     'SaleType']



mssubclass is a categorical data not numerical therefore we can map it to category.


```python
df.MSSubClass = df.MSSubClass.map({20:'class1', 30:'class2', 40:'class3', 45:'class4',
                                   50:'class5', 60:'class6', 70:'class7', 75:'class8',
                                   80:'class9', 85:'class10', 90:'class11', 120:'class12',
                                   150:'class13', 160:'class14', 180:'class15', 190:'class16'})
```


```python
num.remove('MSSubClass')
cat.append('MSSubClass')
```

There are 4 year specifing columns.It would be more helpful if it showed how long back building built or just age of building.


```python
df['Age'] = df.YrSold - df.YearBuilt
df['AgeRemod'] = df.YrSold - df.YearRemodAdd
df['AgeGarage'] = df.YrSold - df.GarageYrBlt
```

there can be -ve values if no garage build.


```python
max_AgeGarage = np.max(df.AgeGarage[df.AgeGarage < 1000])
df['AgeGarage'] = df['AgeGarage'].map(lambda x: max_AgeGarage if x > 1000 else x)

df.Age = df.Age.map(lambda x: 0 if x < 0 else x)
df.AgeRemod = df.AgeRemod.map(lambda x: 0 if x < 0 else x)
df.AgeGarage = df.AgeGarage.map(lambda x: 0 if x < 0 else x)
```


```python
df=df.drop(['YrSold','YearBuilt','YearRemodAdd','GarageYrBlt'],axis=1)

for i in ['YrSold','YearBuilt','YearRemodAdd','GarageYrBlt']: 
    num.remove(i)
num = num + ['Age','AgeRemod','AgeGarage']
```

Create dummy for categorical features that cannot be ranked.


```python
dummy_drop = []
for i in cat:
    dummy_drop += [ i+'_'+str(df[i].unique()[-1]) ]

df = pd.get_dummies(df,columns=cat) 
df = df.drop(dummy_drop,axis=1)
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
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>Alley</th>
      <th>BedroomAbvGr</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>BsmtQual</th>
      <th>BsmtUnfSF</th>
      <th>EnclosedPorch</th>
      <th>ExterCond</th>
      <th>ExterQual</th>
      <th>FireplaceQu</th>
      <th>Fireplaces</th>
      <th>FullBath</th>
      <th>Functional</th>
      <th>GarageArea</th>
      <th>GarageCars</th>
      <th>GarageCond</th>
      <th>GarageQual</th>
      <th>GrLivArea</th>
      <th>HalfBath</th>
      <th>HeatingQC</th>
      <th>Id</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>LandSlope</th>
      <th>LotArea</th>
      <th>LotFrontage</th>
      <th>LowQualFinSF</th>
      <th>MasVnrArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>OpenPorchSF</th>
      <th>OverallCond</th>
      <th>OverallQual</th>
      <th>PavedDrive</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>SalePrice</th>
      <th>ScreenPorch</th>
      <th>Street</th>
      <th>TotRmsAbvGrd</th>
      <th>TotalBsmtSF</th>
      <th>Utilities</th>
      <th>WoodDeckSF</th>
      <th>Age</th>
      <th>AgeRemod</th>
      <th>AgeGarage</th>
      <th>BldgType_1Fam</th>
      <th>BldgType_2fmCon</th>
      <th>BldgType_Duplex</th>
      <th>BldgType_TwnhsE</th>
      <th>CentralAir_Y</th>
      <th>Condition1_Artery</th>
      <th>Condition1_Feedr</th>
      <th>Condition1_Norm</th>
      <th>Condition1_PosA</th>
      <th>Condition1_PosN</th>
      <th>Condition1_RRAe</th>
      <th>Condition1_RRAn</th>
      <th>Condition1_RRNn</th>
      <th>Condition2_Artery</th>
      <th>Condition2_Feedr</th>
      <th>Condition2_Norm</th>
      <th>Condition2_PosA</th>
      <th>Condition2_PosN</th>
      <th>Condition2_RRAn</th>
      <th>Condition2_RRNn</th>
      <th>Electrical_FuseA</th>
      <th>Electrical_FuseF</th>
      <th>Electrical_FuseP</th>
      <th>Electrical_SBrkr</th>
      <th>Exterior1st_AsbShng</th>
      <th>Exterior1st_AsphShn</th>
      <th>Exterior1st_BrkComm</th>
      <th>Exterior1st_BrkFace</th>
      <th>Exterior1st_CemntBd</th>
      <th>Exterior1st_HdBoard</th>
      <th>Exterior1st_ImStucc</th>
      <th>Exterior1st_MetalSd</th>
      <th>Exterior1st_Plywood</th>
      <th>Exterior1st_Stone</th>
      <th>Exterior1st_Stucco</th>
      <th>Exterior1st_VinylSd</th>
      <th>Exterior1st_Wd Sdng</th>
      <th>Exterior1st_WdShing</th>
      <th>Exterior2nd_AsbShng</th>
      <th>Exterior2nd_AsphShn</th>
      <th>Exterior2nd_Brk Cmn</th>
      <th>Exterior2nd_BrkFace</th>
      <th>Exterior2nd_CBlock</th>
      <th>Exterior2nd_CmentBd</th>
      <th>Exterior2nd_HdBoard</th>
      <th>Exterior2nd_ImStucc</th>
      <th>Exterior2nd_MetalSd</th>
      <th>Exterior2nd_Other</th>
      <th>Exterior2nd_Plywood</th>
      <th>Exterior2nd_Stone</th>
      <th>Exterior2nd_Stucco</th>
      <th>Exterior2nd_VinylSd</th>
      <th>Exterior2nd_Wd Sdng</th>
      <th>Exterior2nd_Wd Shng</th>
      <th>Fence_GdPrv</th>
      <th>Fence_GdWo</th>
      <th>Fence_MnPrv</th>
      <th>Fence_NA</th>
      <th>Foundation_BrkTil</th>
      <th>Foundation_CBlock</th>
      <th>Foundation_PConc</th>
      <th>Foundation_Slab</th>
      <th>Foundation_Wood</th>
      <th>GarageFinish_Fin</th>
      <th>GarageFinish_RFn</th>
      <th>GarageFinish_Unf</th>
      <th>GarageType_Attchd</th>
      <th>GarageType_Basment</th>
      <th>GarageType_BuiltIn</th>
      <th>GarageType_CarPort</th>
      <th>GarageType_Detchd</th>
      <th>GarageType_NA</th>
      <th>Heating_GasA</th>
      <th>Heating_GasW</th>
      <th>Heating_Grav</th>
      <th>Heating_OthW</th>
      <th>Heating_Wall</th>
      <th>HouseStyle_1.5Fin</th>
      <th>HouseStyle_1.5Unf</th>
      <th>HouseStyle_1Story</th>
      <th>HouseStyle_2.5Unf</th>
      <th>HouseStyle_2Story</th>
      <th>HouseStyle_SFoyer</th>
      <th>HouseStyle_SLvl</th>
      <th>LandContour_Bnk</th>
      <th>LandContour_Low</th>
      <th>LandContour_Lvl</th>
      <th>LotConfig_Corner</th>
      <th>LotConfig_CulDSac</th>
      <th>LotConfig_FR2</th>
      <th>LotConfig_Inside</th>
      <th>LotShape_IR1</th>
      <th>LotShape_IR2</th>
      <th>LotShape_Reg</th>
      <th>MSZoning_C (all)</th>
      <th>MSZoning_FV</th>
      <th>MSZoning_RL</th>
      <th>MSZoning_RM</th>
      <th>MasVnrType_BrkFace</th>
      <th>MasVnrType_None</th>
      <th>MasVnrType_Stone</th>
      <th>MiscFeature_Gar2</th>
      <th>MiscFeature_NA</th>
      <th>MiscFeature_Othr</th>
      <th>MiscFeature_Shed</th>
      <th>Neighborhood_Blmngtn</th>
      <th>Neighborhood_BrDale</th>
      <th>Neighborhood_BrkSide</th>
      <th>Neighborhood_ClearCr</th>
      <th>Neighborhood_CollgCr</th>
      <th>Neighborhood_Crawfor</th>
      <th>Neighborhood_Edwards</th>
      <th>Neighborhood_Gilbert</th>
      <th>Neighborhood_IDOTRR</th>
      <th>Neighborhood_MeadowV</th>
      <th>Neighborhood_Mitchel</th>
      <th>Neighborhood_NAmes</th>
      <th>Neighborhood_NPkVill</th>
      <th>Neighborhood_NWAmes</th>
      <th>Neighborhood_NoRidge</th>
      <th>Neighborhood_NridgHt</th>
      <th>Neighborhood_OldTown</th>
      <th>Neighborhood_SWISU</th>
      <th>Neighborhood_Sawyer</th>
      <th>Neighborhood_SawyerW</th>
      <th>Neighborhood_Somerst</th>
      <th>Neighborhood_StoneBr</th>
      <th>Neighborhood_Timber</th>
      <th>Neighborhood_Veenker</th>
      <th>RoofMatl_CompShg</th>
      <th>RoofMatl_Membran</th>
      <th>RoofMatl_Metal</th>
      <th>RoofMatl_Roll</th>
      <th>RoofMatl_Tar&amp;Grv</th>
      <th>RoofMatl_WdShake</th>
      <th>RoofMatl_WdShngl</th>
      <th>RoofStyle_Flat</th>
      <th>RoofStyle_Gable</th>
      <th>RoofStyle_Gambrel</th>
      <th>RoofStyle_Hip</th>
      <th>RoofStyle_Mansard</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
      <th>SaleType_COD</th>
      <th>SaleType_CWD</th>
      <th>SaleType_Con</th>
      <th>SaleType_ConLD</th>
      <th>SaleType_ConLI</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_WD</th>
      <th>MSSubClass_class1</th>
      <th>MSSubClass_class10</th>
      <th>MSSubClass_class11</th>
      <th>MSSubClass_class12</th>
      <th>MSSubClass_class14</th>
      <th>MSSubClass_class15</th>
      <th>MSSubClass_class16</th>
      <th>MSSubClass_class2</th>
      <th>MSSubClass_class3</th>
      <th>MSSubClass_class4</th>
      <th>MSSubClass_class5</th>
      <th>MSSubClass_class6</th>
      <th>MSSubClass_class7</th>
      <th>MSSubClass_class8</th>
      <th>MSSubClass_class9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>150.0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>548.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>1710</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>8450</td>
      <td>65.0</td>
      <td>0</td>
      <td>196.0</td>
      <td>0</td>
      <td>2</td>
      <td>61</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>208500.0</td>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>856.0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4</td>
      <td>284.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>460.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>1262</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>9600</td>
      <td>80.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>8</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>181500.0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>1262.0</td>
      <td>4</td>
      <td>298</td>
      <td>31</td>
      <td>31</td>
      <td>31.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>434.0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>608.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>3</td>
      <td>1786</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>11250</td>
      <td>68.0</td>
      <td>0</td>
      <td>162.0</td>
      <td>0</td>
      <td>9</td>
      <td>42</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>223500.0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>920.0</td>
      <td>4</td>
      <td>0</td>
      <td>7</td>
      <td>6</td>
      <td>7.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>540.0</td>
      <td>272</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>642.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>3</td>
      <td>1717</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>9550</td>
      <td>60.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>35</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>140000.0</td>
      <td>0</td>
      <td>2</td>
      <td>7</td>
      <td>756.0</td>
      <td>4</td>
      <td>0</td>
      <td>91</td>
      <td>36</td>
      <td>8.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>490.0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>836.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>3</td>
      <td>2198</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>14260</td>
      <td>84.0</td>
      <td>0</td>
      <td>350.0</td>
      <td>0</td>
      <td>12</td>
      <td>84</td>
      <td>5</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>250000.0</td>
      <td>0</td>
      <td>2</td>
      <td>9</td>
      <td>1145.0</td>
      <td>4</td>
      <td>192</td>
      <td>8</td>
      <td>8</td>
      <td>8.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    1stFlrSF                   int64
    2ndFlrSF                   int64
    3SsnPorch                  int64
    Alley                      int64
    BedroomAbvGr               int64
    BsmtCond                   int64
    BsmtExposure               int64
    BsmtFinSF1               float64
    BsmtFinSF2               float64
    BsmtFinType1               int64
    BsmtFinType2               int64
    BsmtFullBath             float64
    BsmtHalfBath             float64
    BsmtQual                   int64
    BsmtUnfSF                float64
    EnclosedPorch              int64
    ExterCond                  int64
    ExterQual                  int64
    FireplaceQu                int64
    Fireplaces                 int64
    FullBath                   int64
    Functional                 int64
    GarageArea               float64
    GarageCars               float64
    GarageCond                 int64
    GarageQual                 int64
    GrLivArea                  int64
    HalfBath                   int64
    HeatingQC                  int64
    Id                         int64
    KitchenAbvGr               int64
    KitchenQual                int64
    LandSlope                  int64
    LotArea                    int64
    LotFrontage              float64
    LowQualFinSF               int64
    MasVnrArea               float64
    MiscVal                    int64
    MoSold                     int64
    OpenPorchSF                int64
    OverallCond                int64
    OverallQual                int64
    PavedDrive                 int64
    PoolArea                   int64
    PoolQC                     int64
    SalePrice                float64
    ScreenPorch                int64
    Street                     int64
    TotRmsAbvGrd               int64
    TotalBsmtSF              float64
    Utilities                  int64
    WoodDeckSF                 int64
    Age                        int64
    AgeRemod                   int64
    AgeGarage                float64
    BldgType_1Fam              uint8
    BldgType_2fmCon            uint8
    BldgType_Duplex            uint8
    BldgType_TwnhsE            uint8
    CentralAir_Y               uint8
    Condition1_Artery          uint8
    Condition1_Feedr           uint8
    Condition1_Norm            uint8
    Condition1_PosA            uint8
    Condition1_PosN            uint8
    Condition1_RRAe            uint8
    Condition1_RRAn            uint8
    Condition1_RRNn            uint8
    Condition2_Artery          uint8
    Condition2_Feedr           uint8
    Condition2_Norm            uint8
    Condition2_PosA            uint8
    Condition2_PosN            uint8
    Condition2_RRAn            uint8
    Condition2_RRNn            uint8
    Electrical_FuseA           uint8
    Electrical_FuseF           uint8
    Electrical_FuseP           uint8
    Electrical_SBrkr           uint8
    Exterior1st_AsbShng        uint8
    Exterior1st_AsphShn        uint8
    Exterior1st_BrkComm        uint8
    Exterior1st_BrkFace        uint8
    Exterior1st_CemntBd        uint8
    Exterior1st_HdBoard        uint8
    Exterior1st_ImStucc        uint8
    Exterior1st_MetalSd        uint8
    Exterior1st_Plywood        uint8
    Exterior1st_Stone          uint8
    Exterior1st_Stucco         uint8
    Exterior1st_VinylSd        uint8
    Exterior1st_Wd Sdng        uint8
    Exterior1st_WdShing        uint8
    Exterior2nd_AsbShng        uint8
    Exterior2nd_AsphShn        uint8
    Exterior2nd_Brk Cmn        uint8
    Exterior2nd_BrkFace        uint8
    Exterior2nd_CBlock         uint8
    Exterior2nd_CmentBd        uint8
    Exterior2nd_HdBoard        uint8
    Exterior2nd_ImStucc        uint8
    Exterior2nd_MetalSd        uint8
    Exterior2nd_Other          uint8
    Exterior2nd_Plywood        uint8
    Exterior2nd_Stone          uint8
    Exterior2nd_Stucco         uint8
    Exterior2nd_VinylSd        uint8
    Exterior2nd_Wd Sdng        uint8
    Exterior2nd_Wd Shng        uint8
    Fence_GdPrv                uint8
    Fence_GdWo                 uint8
    Fence_MnPrv                uint8
    Fence_NA                   uint8
    Foundation_BrkTil          uint8
    Foundation_CBlock          uint8
    Foundation_PConc           uint8
    Foundation_Slab            uint8
    Foundation_Wood            uint8
    GarageFinish_Fin           uint8
    GarageFinish_RFn           uint8
    GarageFinish_Unf           uint8
    GarageType_Attchd          uint8
    GarageType_Basment         uint8
    GarageType_BuiltIn         uint8
    GarageType_CarPort         uint8
    GarageType_Detchd          uint8
    GarageType_NA              uint8
    Heating_GasA               uint8
    Heating_GasW               uint8
    Heating_Grav               uint8
    Heating_OthW               uint8
    Heating_Wall               uint8
    HouseStyle_1.5Fin          uint8
    HouseStyle_1.5Unf          uint8
    HouseStyle_1Story          uint8
    HouseStyle_2.5Unf          uint8
    HouseStyle_2Story          uint8
    HouseStyle_SFoyer          uint8
    HouseStyle_SLvl            uint8
    LandContour_Bnk            uint8
    LandContour_Low            uint8
    LandContour_Lvl            uint8
    LotConfig_Corner           uint8
    LotConfig_CulDSac          uint8
    LotConfig_FR2              uint8
    LotConfig_Inside           uint8
    LotShape_IR1               uint8
    LotShape_IR2               uint8
    LotShape_Reg               uint8
    MSZoning_C (all)           uint8
    MSZoning_FV                uint8
    MSZoning_RL                uint8
    MSZoning_RM                uint8
    MasVnrType_BrkFace         uint8
    MasVnrType_None            uint8
    MasVnrType_Stone           uint8
    MiscFeature_Gar2           uint8
    MiscFeature_NA             uint8
    MiscFeature_Othr           uint8
    MiscFeature_Shed           uint8
    Neighborhood_Blmngtn       uint8
    Neighborhood_BrDale        uint8
    Neighborhood_BrkSide       uint8
    Neighborhood_ClearCr       uint8
    Neighborhood_CollgCr       uint8
    Neighborhood_Crawfor       uint8
    Neighborhood_Edwards       uint8
    Neighborhood_Gilbert       uint8
    Neighborhood_IDOTRR        uint8
    Neighborhood_MeadowV       uint8
    Neighborhood_Mitchel       uint8
    Neighborhood_NAmes         uint8
    Neighborhood_NPkVill       uint8
    Neighborhood_NWAmes        uint8
    Neighborhood_NoRidge       uint8
    Neighborhood_NridgHt       uint8
    Neighborhood_OldTown       uint8
    Neighborhood_SWISU         uint8
    Neighborhood_Sawyer        uint8
    Neighborhood_SawyerW       uint8
    Neighborhood_Somerst       uint8
    Neighborhood_StoneBr       uint8
    Neighborhood_Timber        uint8
    Neighborhood_Veenker       uint8
    RoofMatl_CompShg           uint8
    RoofMatl_Membran           uint8
    RoofMatl_Metal             uint8
    RoofMatl_Roll              uint8
    RoofMatl_Tar&Grv           uint8
    RoofMatl_WdShake           uint8
    RoofMatl_WdShngl           uint8
    RoofStyle_Flat             uint8
    RoofStyle_Gable            uint8
    RoofStyle_Gambrel          uint8
    RoofStyle_Hip              uint8
    RoofStyle_Mansard          uint8
    SaleCondition_Abnorml      uint8
    SaleCondition_AdjLand      uint8
    SaleCondition_Alloca       uint8
    SaleCondition_Normal       uint8
    SaleCondition_Partial      uint8
    SaleType_COD               uint8
    SaleType_CWD               uint8
    SaleType_Con               uint8
    SaleType_ConLD             uint8
    SaleType_ConLI             uint8
    SaleType_ConLw             uint8
    SaleType_New               uint8
    SaleType_WD                uint8
    MSSubClass_class1          uint8
    MSSubClass_class10         uint8
    MSSubClass_class11         uint8
    MSSubClass_class12         uint8
    MSSubClass_class14         uint8
    MSSubClass_class15         uint8
    MSSubClass_class16         uint8
    MSSubClass_class2          uint8
    MSSubClass_class3          uint8
    MSSubClass_class4          uint8
    MSSubClass_class5          uint8
    MSSubClass_class6          uint8
    MSSubClass_class7          uint8
    MSSubClass_class8          uint8
    MSSubClass_class9          uint8
    dtype: object




```python
df.shape
```




    (2919, 224)



Split dataset train as first 1460 rows and test as last 1459 rows because we appended a train and test earilier.


```python
X_train  = df[:-1459].drop(['SalePrice','Id'], axis=1)
y_train  = df[:-1459]['SalePrice']
X_test  = df[-1459:].drop(['SalePrice','Id'], axis=1)
```


```python
from xgboost import XGBRegressor
```


```python
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
```




    XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints=None,
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints=None,
                 n_estimators=100, n_jobs=0, num_parallel_tree=1,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
                 validate_parameters=False, verbosity=None)




```python
xgb.score(X_train, y_train)
```

we have to take only few features as to improve model.


```python
imp = pd.DataFrame(xgb.feature_importances_ ,columns = ['Importance'],index = X_train.columns)
imp = imp.sort_values(['Importance'], ascending = False)
```


```python
imp
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
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OverallQual</td>
      <td>0.446473</td>
    </tr>
    <tr>
      <td>GarageCars</td>
      <td>0.135958</td>
    </tr>
    <tr>
      <td>CentralAir_Y</td>
      <td>0.059187</td>
    </tr>
    <tr>
      <td>GrLivArea</td>
      <td>0.028329</td>
    </tr>
    <tr>
      <td>MSZoning_RM</td>
      <td>0.027766</td>
    </tr>
    <tr>
      <td>KitchenAbvGr</td>
      <td>0.020193</td>
    </tr>
    <tr>
      <td>TotalBsmtSF</td>
      <td>0.020160</td>
    </tr>
    <tr>
      <td>RoofMatl_CompShg</td>
      <td>0.017258</td>
    </tr>
    <tr>
      <td>BsmtQual</td>
      <td>0.016712</td>
    </tr>
    <tr>
      <td>GarageQual</td>
      <td>0.015863</td>
    </tr>
    <tr>
      <td>KitchenQual</td>
      <td>0.015859</td>
    </tr>
    <tr>
      <td>FullBath</td>
      <td>0.011979</td>
    </tr>
    <tr>
      <td>MSZoning_RL</td>
      <td>0.010424</td>
    </tr>
    <tr>
      <td>Alley</td>
      <td>0.007786</td>
    </tr>
    <tr>
      <td>GarageType_Attchd</td>
      <td>0.007374</td>
    </tr>
    <tr>
      <td>LandSlope</td>
      <td>0.007261</td>
    </tr>
    <tr>
      <td>Neighborhood_Crawfor</td>
      <td>0.007020</td>
    </tr>
    <tr>
      <td>Heating_Grav</td>
      <td>0.005074</td>
    </tr>
    <tr>
      <td>BsmtFinSF1</td>
      <td>0.004636</td>
    </tr>
    <tr>
      <td>RoofStyle_Flat</td>
      <td>0.004517</td>
    </tr>
    <tr>
      <td>ExterQual</td>
      <td>0.004195</td>
    </tr>
    <tr>
      <td>LandContour_Bnk</td>
      <td>0.004156</td>
    </tr>
    <tr>
      <td>OverallCond</td>
      <td>0.003864</td>
    </tr>
    <tr>
      <td>Condition2_Norm</td>
      <td>0.003703</td>
    </tr>
    <tr>
      <td>MSZoning_C (all)</td>
      <td>0.003549</td>
    </tr>
    <tr>
      <td>AgeRemod</td>
      <td>0.003455</td>
    </tr>
    <tr>
      <td>SaleType_WD</td>
      <td>0.003165</td>
    </tr>
    <tr>
      <td>Condition1_PosA</td>
      <td>0.003120</td>
    </tr>
    <tr>
      <td>1stFlrSF</td>
      <td>0.003047</td>
    </tr>
    <tr>
      <td>Exterior1st_HdBoard</td>
      <td>0.002962</td>
    </tr>
    <tr>
      <td>HouseStyle_1.5Fin</td>
      <td>0.002960</td>
    </tr>
    <tr>
      <td>FireplaceQu</td>
      <td>0.002856</td>
    </tr>
    <tr>
      <td>GarageArea</td>
      <td>0.002603</td>
    </tr>
    <tr>
      <td>BedroomAbvGr</td>
      <td>0.002332</td>
    </tr>
    <tr>
      <td>Functional</td>
      <td>0.002304</td>
    </tr>
    <tr>
      <td>GarageCond</td>
      <td>0.002291</td>
    </tr>
    <tr>
      <td>Neighborhood_Somerst</td>
      <td>0.002289</td>
    </tr>
    <tr>
      <td>Exterior1st_BrkFace</td>
      <td>0.002283</td>
    </tr>
    <tr>
      <td>Age</td>
      <td>0.002091</td>
    </tr>
    <tr>
      <td>Neighborhood_StoneBr</td>
      <td>0.001960</td>
    </tr>
    <tr>
      <td>2ndFlrSF</td>
      <td>0.001952</td>
    </tr>
    <tr>
      <td>MSZoning_FV</td>
      <td>0.001828</td>
    </tr>
    <tr>
      <td>LotConfig_CulDSac</td>
      <td>0.001780</td>
    </tr>
    <tr>
      <td>Neighborhood_ClearCr</td>
      <td>0.001741</td>
    </tr>
    <tr>
      <td>ExterCond</td>
      <td>0.001660</td>
    </tr>
    <tr>
      <td>LotArea</td>
      <td>0.001658</td>
    </tr>
    <tr>
      <td>BsmtFinSF2</td>
      <td>0.001631</td>
    </tr>
    <tr>
      <td>Exterior2nd_Wd Shng</td>
      <td>0.001592</td>
    </tr>
    <tr>
      <td>BsmtExposure</td>
      <td>0.001530</td>
    </tr>
    <tr>
      <td>Fence_GdPrv</td>
      <td>0.001515</td>
    </tr>
    <tr>
      <td>Exterior1st_MetalSd</td>
      <td>0.001493</td>
    </tr>
    <tr>
      <td>TotRmsAbvGrd</td>
      <td>0.001474</td>
    </tr>
    <tr>
      <td>BsmtFinType1</td>
      <td>0.001448</td>
    </tr>
    <tr>
      <td>SaleCondition_Abnorml</td>
      <td>0.001445</td>
    </tr>
    <tr>
      <td>MSSubClass_class2</td>
      <td>0.001369</td>
    </tr>
    <tr>
      <td>PoolArea</td>
      <td>0.001332</td>
    </tr>
    <tr>
      <td>OpenPorchSF</td>
      <td>0.001320</td>
    </tr>
    <tr>
      <td>RoofStyle_Gable</td>
      <td>0.001304</td>
    </tr>
    <tr>
      <td>Neighborhood_Mitchel</td>
      <td>0.001283</td>
    </tr>
    <tr>
      <td>AgeGarage</td>
      <td>0.001261</td>
    </tr>
    <tr>
      <td>ScreenPorch</td>
      <td>0.001214</td>
    </tr>
    <tr>
      <td>BsmtFullBath</td>
      <td>0.001144</td>
    </tr>
    <tr>
      <td>Condition1_Artery</td>
      <td>0.001122</td>
    </tr>
    <tr>
      <td>BsmtCond</td>
      <td>0.001035</td>
    </tr>
    <tr>
      <td>HouseStyle_SLvl</td>
      <td>0.001021</td>
    </tr>
    <tr>
      <td>Neighborhood_Edwards</td>
      <td>0.000935</td>
    </tr>
    <tr>
      <td>SaleCondition_Partial</td>
      <td>0.000894</td>
    </tr>
    <tr>
      <td>Fireplaces</td>
      <td>0.000888</td>
    </tr>
    <tr>
      <td>LotFrontage</td>
      <td>0.000866</td>
    </tr>
    <tr>
      <td>Neighborhood_NridgHt</td>
      <td>0.000842</td>
    </tr>
    <tr>
      <td>Neighborhood_CollgCr</td>
      <td>0.000815</td>
    </tr>
    <tr>
      <td>PavedDrive</td>
      <td>0.000809</td>
    </tr>
    <tr>
      <td>LotShape_Reg</td>
      <td>0.000778</td>
    </tr>
    <tr>
      <td>Neighborhood_Timber</td>
      <td>0.000716</td>
    </tr>
    <tr>
      <td>Exterior1st_Plywood</td>
      <td>0.000715</td>
    </tr>
    <tr>
      <td>WoodDeckSF</td>
      <td>0.000711</td>
    </tr>
    <tr>
      <td>HouseStyle_2Story</td>
      <td>0.000668</td>
    </tr>
    <tr>
      <td>Neighborhood_SWISU</td>
      <td>0.000665</td>
    </tr>
    <tr>
      <td>LotConfig_Inside</td>
      <td>0.000644</td>
    </tr>
    <tr>
      <td>Condition2_Feedr</td>
      <td>0.000625</td>
    </tr>
    <tr>
      <td>LotShape_IR1</td>
      <td>0.000583</td>
    </tr>
    <tr>
      <td>Neighborhood_Sawyer</td>
      <td>0.000576</td>
    </tr>
    <tr>
      <td>LotConfig_FR2</td>
      <td>0.000575</td>
    </tr>
    <tr>
      <td>Electrical_SBrkr</td>
      <td>0.000566</td>
    </tr>
    <tr>
      <td>RoofStyle_Hip</td>
      <td>0.000565</td>
    </tr>
    <tr>
      <td>MasVnrType_Stone</td>
      <td>0.000551</td>
    </tr>
    <tr>
      <td>GarageType_Detchd</td>
      <td>0.000516</td>
    </tr>
    <tr>
      <td>Exterior1st_Wd Sdng</td>
      <td>0.000513</td>
    </tr>
    <tr>
      <td>MSSubClass_class1</td>
      <td>0.000509</td>
    </tr>
    <tr>
      <td>GarageFinish_Fin</td>
      <td>0.000499</td>
    </tr>
    <tr>
      <td>Condition1_RRNn</td>
      <td>0.000496</td>
    </tr>
    <tr>
      <td>Neighborhood_BrkSide</td>
      <td>0.000487</td>
    </tr>
    <tr>
      <td>EnclosedPorch</td>
      <td>0.000481</td>
    </tr>
    <tr>
      <td>MSSubClass_class12</td>
      <td>0.000480</td>
    </tr>
    <tr>
      <td>GarageFinish_RFn</td>
      <td>0.000473</td>
    </tr>
    <tr>
      <td>Neighborhood_NAmes</td>
      <td>0.000468</td>
    </tr>
    <tr>
      <td>Exterior1st_AsbShng</td>
      <td>0.000467</td>
    </tr>
    <tr>
      <td>Fence_MnPrv</td>
      <td>0.000459</td>
    </tr>
    <tr>
      <td>MSSubClass_class6</td>
      <td>0.000458</td>
    </tr>
    <tr>
      <td>MasVnrArea</td>
      <td>0.000439</td>
    </tr>
    <tr>
      <td>BsmtUnfSF</td>
      <td>0.000438</td>
    </tr>
    <tr>
      <td>Neighborhood_Gilbert</td>
      <td>0.000433</td>
    </tr>
    <tr>
      <td>HeatingQC</td>
      <td>0.000420</td>
    </tr>
    <tr>
      <td>SaleCondition_AdjLand</td>
      <td>0.000404</td>
    </tr>
    <tr>
      <td>Condition1_Norm</td>
      <td>0.000394</td>
    </tr>
    <tr>
      <td>MoSold</td>
      <td>0.000393</td>
    </tr>
    <tr>
      <td>LotConfig_Corner</td>
      <td>0.000393</td>
    </tr>
    <tr>
      <td>SaleType_New</td>
      <td>0.000385</td>
    </tr>
    <tr>
      <td>Foundation_CBlock</td>
      <td>0.000377</td>
    </tr>
    <tr>
      <td>Condition1_RRAe</td>
      <td>0.000362</td>
    </tr>
    <tr>
      <td>BsmtHalfBath</td>
      <td>0.000335</td>
    </tr>
    <tr>
      <td>SaleCondition_Normal</td>
      <td>0.000333</td>
    </tr>
    <tr>
      <td>Fence_NA</td>
      <td>0.000322</td>
    </tr>
    <tr>
      <td>Neighborhood_OldTown</td>
      <td>0.000321</td>
    </tr>
    <tr>
      <td>MasVnrType_None</td>
      <td>0.000312</td>
    </tr>
    <tr>
      <td>SaleType_ConLI</td>
      <td>0.000299</td>
    </tr>
    <tr>
      <td>SaleType_COD</td>
      <td>0.000288</td>
    </tr>
    <tr>
      <td>Neighborhood_NoRidge</td>
      <td>0.000265</td>
    </tr>
    <tr>
      <td>Exterior1st_Stucco</td>
      <td>0.000259</td>
    </tr>
    <tr>
      <td>MiscVal</td>
      <td>0.000233</td>
    </tr>
    <tr>
      <td>Exterior1st_VinylSd</td>
      <td>0.000231</td>
    </tr>
    <tr>
      <td>Exterior2nd_VinylSd</td>
      <td>0.000221</td>
    </tr>
    <tr>
      <td>LandContour_Low</td>
      <td>0.000219</td>
    </tr>
    <tr>
      <td>MSSubClass_class3</td>
      <td>0.000215</td>
    </tr>
    <tr>
      <td>LandContour_Lvl</td>
      <td>0.000213</td>
    </tr>
    <tr>
      <td>HalfBath</td>
      <td>0.000202</td>
    </tr>
    <tr>
      <td>Fence_GdWo</td>
      <td>0.000201</td>
    </tr>
    <tr>
      <td>MSSubClass_class9</td>
      <td>0.000196</td>
    </tr>
    <tr>
      <td>Neighborhood_NWAmes</td>
      <td>0.000194</td>
    </tr>
    <tr>
      <td>Exterior2nd_Stucco</td>
      <td>0.000185</td>
    </tr>
    <tr>
      <td>LotShape_IR2</td>
      <td>0.000181</td>
    </tr>
    <tr>
      <td>HouseStyle_1Story</td>
      <td>0.000178</td>
    </tr>
    <tr>
      <td>RoofMatl_WdShngl</td>
      <td>0.000166</td>
    </tr>
    <tr>
      <td>GarageType_CarPort</td>
      <td>0.000166</td>
    </tr>
    <tr>
      <td>MSSubClass_class5</td>
      <td>0.000163</td>
    </tr>
    <tr>
      <td>Neighborhood_MeadowV</td>
      <td>0.000159</td>
    </tr>
    <tr>
      <td>Electrical_FuseA</td>
      <td>0.000158</td>
    </tr>
    <tr>
      <td>Exterior2nd_HdBoard</td>
      <td>0.000157</td>
    </tr>
    <tr>
      <td>BldgType_2fmCon</td>
      <td>0.000149</td>
    </tr>
    <tr>
      <td>3SsnPorch</td>
      <td>0.000149</td>
    </tr>
    <tr>
      <td>Exterior2nd_Plywood</td>
      <td>0.000147</td>
    </tr>
    <tr>
      <td>BsmtFinType2</td>
      <td>0.000142</td>
    </tr>
    <tr>
      <td>BldgType_1Fam</td>
      <td>0.000138</td>
    </tr>
    <tr>
      <td>GarageType_BuiltIn</td>
      <td>0.000125</td>
    </tr>
    <tr>
      <td>LowQualFinSF</td>
      <td>0.000122</td>
    </tr>
    <tr>
      <td>Neighborhood_Blmngtn</td>
      <td>0.000106</td>
    </tr>
    <tr>
      <td>MSSubClass_class7</td>
      <td>0.000102</td>
    </tr>
    <tr>
      <td>Exterior2nd_CmentBd</td>
      <td>0.000100</td>
    </tr>
    <tr>
      <td>MasVnrType_BrkFace</td>
      <td>0.000099</td>
    </tr>
    <tr>
      <td>Condition1_PosN</td>
      <td>0.000098</td>
    </tr>
    <tr>
      <td>Neighborhood_SawyerW</td>
      <td>0.000096</td>
    </tr>
    <tr>
      <td>HouseStyle_SFoyer</td>
      <td>0.000091</td>
    </tr>
    <tr>
      <td>GarageFinish_Unf</td>
      <td>0.000089</td>
    </tr>
    <tr>
      <td>Condition1_Feedr</td>
      <td>0.000081</td>
    </tr>
    <tr>
      <td>Electrical_FuseF</td>
      <td>0.000075</td>
    </tr>
    <tr>
      <td>Foundation_PConc</td>
      <td>0.000073</td>
    </tr>
    <tr>
      <td>BldgType_TwnhsE</td>
      <td>0.000070</td>
    </tr>
    <tr>
      <td>MSSubClass_class10</td>
      <td>0.000070</td>
    </tr>
    <tr>
      <td>Neighborhood_BrDale</td>
      <td>0.000067</td>
    </tr>
    <tr>
      <td>Condition1_RRAn</td>
      <td>0.000064</td>
    </tr>
    <tr>
      <td>Exterior2nd_ImStucc</td>
      <td>0.000058</td>
    </tr>
    <tr>
      <td>Neighborhood_IDOTRR</td>
      <td>0.000056</td>
    </tr>
    <tr>
      <td>Exterior1st_CemntBd</td>
      <td>0.000054</td>
    </tr>
    <tr>
      <td>Exterior2nd_MetalSd</td>
      <td>0.000045</td>
    </tr>
    <tr>
      <td>MiscFeature_Othr</td>
      <td>0.000044</td>
    </tr>
    <tr>
      <td>Exterior1st_WdShing</td>
      <td>0.000043</td>
    </tr>
    <tr>
      <td>Exterior2nd_Wd Sdng</td>
      <td>0.000042</td>
    </tr>
    <tr>
      <td>Exterior1st_AsphShn</td>
      <td>0.000040</td>
    </tr>
    <tr>
      <td>Foundation_Slab</td>
      <td>0.000040</td>
    </tr>
    <tr>
      <td>Exterior2nd_Stone</td>
      <td>0.000030</td>
    </tr>
    <tr>
      <td>SaleType_Con</td>
      <td>0.000017</td>
    </tr>
    <tr>
      <td>Foundation_BrkTil</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <td>Exterior1st_ImStucc</td>
      <td>0.000004</td>
    </tr>
    <tr>
      <td>MSSubClass_class11</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>BldgType_Duplex</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>SaleType_CWD</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MiscFeature_Gar2</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>SaleType_ConLD</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior2nd_Other</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>SaleType_ConLw</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Foundation_Wood</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Utilities</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Street</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>GarageType_Basment</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>PoolQC</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Neighborhood_NPkVill</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MSSubClass_class14</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MSSubClass_class15</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MSSubClass_class16</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>GarageType_NA</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Heating_GasA</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MSSubClass_class4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>HouseStyle_2.5Unf</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>HouseStyle_1.5Unf</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Heating_GasW</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MSSubClass_class8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MiscFeature_NA</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>MiscFeature_Shed</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>SaleCondition_Alloca</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior2nd_CBlock</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior1st_BrkComm</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Heating_Wall</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Electrical_FuseP</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior1st_Stone</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior2nd_AsbShng</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior2nd_AsphShn</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Condition2_RRNn</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Neighborhood_Veenker</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Condition2_RRAn</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofMatl_Membran</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofMatl_Metal</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofMatl_Roll</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofMatl_Tar&amp;Grv</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofMatl_WdShake</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior2nd_Brk Cmn</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Condition2_PosN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Condition2_PosA</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofStyle_Gambrel</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Condition2_Artery</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>RoofStyle_Mansard</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Exterior2nd_BrkFace</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Heating_OthW</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
feat_sel = imp[:56]
```


```python
feat_list = feat_sel.index.tolist
```


```python
feat_list()
```




    ['OverallQual',
     'GarageCars',
     'CentralAir_Y',
     'GrLivArea',
     'MSZoning_RM',
     'KitchenAbvGr',
     'TotalBsmtSF',
     'RoofMatl_CompShg',
     'BsmtQual',
     'GarageQual',
     'KitchenQual',
     'FullBath',
     'MSZoning_RL',
     'Alley',
     'GarageType_Attchd',
     'LandSlope',
     'Neighborhood_Crawfor',
     'Heating_Grav',
     'BsmtFinSF1',
     'RoofStyle_Flat',
     'ExterQual',
     'LandContour_Bnk',
     'OverallCond',
     'Condition2_Norm',
     'MSZoning_C (all)',
     'AgeRemod',
     'SaleType_WD',
     'Condition1_PosA',
     '1stFlrSF',
     'Exterior1st_HdBoard',
     'HouseStyle_1.5Fin',
     'FireplaceQu',
     'GarageArea',
     'BedroomAbvGr',
     'Functional',
     'GarageCond',
     'Neighborhood_Somerst',
     'Exterior1st_BrkFace',
     'Age',
     'Neighborhood_StoneBr',
     '2ndFlrSF',
     'MSZoning_FV',
     'LotConfig_CulDSac',
     'Neighborhood_ClearCr',
     'ExterCond',
     'LotArea',
     'BsmtFinSF2',
     'Exterior2nd_Wd Shng',
     'BsmtExposure',
     'Fence_GdPrv',
     'Exterior1st_MetalSd',
     'TotRmsAbvGrd',
     'BsmtFinType1',
     'SaleCondition_Abnorml',
     'MSSubClass_class2',
     'PoolArea']




```python
df_new = df.copy()
```


```python
df_new = df_new.filter(['OverallQual', 'GarageCars', 'CentralAir_Y', 'GrLivArea', 'MSZoning_RM', 'KitchenAbvGr', 'TotalBsmtSF', 
                  'BsmtQual', 'GarageQual', 'KitchenQual', 'FullBath', 'RoofMatl_CompShg', 'MSZoning_RL', 'Alley', 
                  'GarageType_Attchd', 'LandSlope', 'Neighborhood_Crawfor', 'Condition1_PosA', 'HouseStyle_1.5Fin', 
                  'Heating_Grav', 'BsmtFinSF1', 'RoofStyle_Flat', 'ExterQual', 'OverallCond', 'Condition2_Norm', 
                  'MSZoning_C (all)', 'AgeRemod', '1stFlrSF', 'Exterior1st_HdBoard', 'FireplaceQu', 'LandContour_Bnk', 
                  'Neighborhood_StoneBr', 'SaleType_WD', 'GarageArea', 'BedroomAbvGr', 'Functional', 'GarageCond', 
                  'Neighborhood_Somerst', 'Exterior1st_BrkFace', 'Age', '2ndFlrSF', 'MSZoning_FV', 'LotConfig_CulDSac', 
                  'Neighborhood_ClearCr', 'ExterCond', 'LotArea', 'BsmtFinSF2', 'Exterior2nd_Wd Shng', 'BsmtExposure', 
                  'Fence_GdPrv', 'TotRmsAbvGrd', 'BsmtFinType1', 'SaleCondition_Abnorml', 'MSSubClass_class2', 
                  'PoolArea', 'OpenPorchSF','SalePrice'])
```


```python
X_train  = df_new[:-1459].drop(['SalePrice'], axis=1)
y_train  = df_new[:-1459]['SalePrice']
X_test  = df_new[-1459:].drop(['SalePrice'], axis=1)
```


```python
X_train.shape,y_train.shape,X_test.shape
```




    ((1460, 56), (1460,), (1459, 56))




```python
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
```




    XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints=None,
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints=None,
                 n_estimators=100, n_jobs=0, num_parallel_tree=1,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
                 validate_parameters=False, verbosity=None)




```python
xgb.score(X_train, y_train)
```




    0.9992116339685525




```python
y_pred = xgb.predict(X_test)
```


```python
testID = pd.read_csv('test.csv')
```


```python
output = pd.DataFrame({'Id': testID['Id'], 'SalePrice': y_pred})
output.to_csv('predictionfinal.csv', index=False)
```


```python

```
