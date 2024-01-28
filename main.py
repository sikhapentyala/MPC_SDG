import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

path = "/Users/sikha/Development/MPC_SDG/private-pgm-master/data/"
#cols = ['class','lymphatics','block of affere','bl. of lymph.','bl. of lymph. s','by pass','extravasates','regeneration of','early uptake in','lym.nodes dimin','lym.nodes enlar','changes in lym.','defect in node','changes in node','changes in stru','special forms','dislocation of','exclusion of no','no. of nodes in']
cols = ['class','age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad','irradiat']
#df = pd.read_csv('/Users/sikha/Development/MPC_SDG/private-pgm-master/data/breast-cancer.csv')
#df.columns = cols
#df_test = pd.read_csv('/Users/sikha/Development/MPC_SDG/private-pgm-master/data/compas_test.csv')
#df.to_csv("/Users/sikha/Development/MPC_SDG/private-pgm-master/data/breast-cancer.csv")

df = pd.read_csv('/Users/sikha/Development/MPC_SDG/private-pgm-master/data/breast-cancer.csv')
cols_ = df[cols].columns
ordinal_encoder = OrdinalEncoder()
encoded_data = ordinal_encoder.fit_transform(df[cols])
df_encoded = pd.DataFrame(encoded_data, columns=cols)
print(df_encoded.head())
df_encoded.to_csv("/Users/sikha/Development/MPC_SDG/private-pgm-master/data/breast-cancer_enc.csv")
#df = df_train.append(df_test)
#print(df.columns)
#print(df.nunique())

domain_df = {}
for c in df_encoded.columns:
    print(c)
    print(len(df_encoded[c].value_counts()))
    domain_df[c] = len(df_encoded[c].value_counts())
#    print('************')
print(domain_df)


