import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

df=pd.read_csv("data/customers.csv")
X=df[["Age","AnnualIncome","SpendingScore"]]

scaler=StandardScaler()
Xs=scaler.fit_transform(X)

wcss=[]
for k in range(1,11):
    wcss.append(KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xs).inertia_)
plt.plot(range(1,11),wcss,marker="o")
plt.title("Elbow Method")
plt.savefig("reports/elbow.png")
plt.close()

model=KMeans(n_clusters=4,n_init=10,random_state=42)
df["Cluster"]=model.fit_predict(Xs)

pca=PCA(n_components=2)
pts=pca.fit_transform(Xs)
plt.scatter(pts[:,0],pts[:,1],c=df["Cluster"])
plt.title("Customer Segments")
plt.savefig("reports/customer_segments.png")
joblib.dump(model,"models/kmeans.pkl")
print(df.groupby("Cluster").mean())
