import pandas as pd

df = pd.read_csv('Code/Dataset/FinalCleanDataset.csv')

ion_name = dict.fromkeys(df['Working Ion'].unique(), 0)
print(df.shape)
for value in df['Working Ion']:
    if value in ion_name:
        ion_name[value] += 1
    
for key in ion_name:
    print("Composition of", key, "is", (ion_name[key]/6684)*100.00,"%")