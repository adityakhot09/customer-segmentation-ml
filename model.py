import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

def segment_customers():
    # -----------------------------
    # 1️⃣ Read CSV
    # -----------------------------
    data = pd.read_csv("customers.csv")
    data["PhoneNumber"] = data["PhoneNumber"].astype(str)

    # -----------------------------
    # 2️⃣ Group Same Customers
    # -----------------------------
    customer_data = data.groupby("PhoneNumber").agg({
        "Name": "first",
        "Age": "first",
        "PurchaseAmount": ["sum", "count"]
    }).reset_index()

    customer_data.columns = [
        "PhoneNumber",
        "Name",
        "Age",
        "TotalSpend",
        "PurchaseFrequency"
    ]

    # -----------------------------
    # 3️⃣ Monthly Cost
    # -----------------------------
    customer_data["MonthlyCost"] = (
        customer_data["TotalSpend"] / 12
    ).round(2)

    # -----------------------------
    # 4️⃣ Spending Score (1–100)
    # -----------------------------
    scaler = MinMaxScaler(feature_range=(1, 100))
    customer_data["SpendingScore"] = scaler.fit_transform(
        customer_data[["TotalSpend"]]
    ).round(0)

    # -----------------------------
    # 5️⃣ K-Means Clustering
    # -----------------------------
    X = customer_data[["SpendingScore", "PurchaseFrequency"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data["Cluster"] = kmeans.fit_predict(X)

    # -----------------------------
    # 6️⃣ Assign Labels
    # -----------------------------
    cluster_means = (
        customer_data.groupby("Cluster")["SpendingScore"]
        .mean()
        .sort_values()
    )

    labels = ["Low Spenders", "Medium Spenders", "High Spenders"]
    label_map = {
        cluster: labels[i]
        for i, cluster in enumerate(cluster_means.index)
    }

    customer_data["CustomerGroup"] = customer_data["Cluster"].map(label_map)

    # -----------------------------
    # 7️⃣ SORTING (🔥 IMPORTANT)
    # -----------------------------
    group_order = {
        "Low Spenders": 0,
        "Medium Spenders": 1,
        "High Spenders": 2
    }

    customer_data["GroupOrder"] = customer_data["CustomerGroup"].map(group_order)

    customer_data = customer_data.sort_values(
        by=["GroupOrder", "TotalSpend", "PurchaseFrequency"],
        ascending=[True, True, False]
    )

    # -----------------------------
    # 8️⃣ Save Cluster Graph
    # -----------------------------
    os.makedirs("static", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.scatter(
        customer_data["PurchaseFrequency"],
        customer_data["SpendingScore"],
        c=customer_data["Cluster"]
    )
    plt.xlabel("Purchase Frequency")
    plt.ylabel("Spending Score")
    plt.title("Customer Segmentation (K-Means)")
    plt.savefig("static/kmeans_clusters.png")
    plt.close()

    # -----------------------------
    # 9️⃣ Final Output
    # -----------------------------
    customer_data = customer_data.drop(columns=["Cluster", "GroupOrder"])

    return customer_data.to_dict(orient="records")
