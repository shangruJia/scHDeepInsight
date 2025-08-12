import scanpy as sc
import celltypist
from celltypist import models
import pandas as pd

# Load data
query = sc.read('./path/to/data.h5ad')

# Ensure gene names are properly set
if "feature_name" not in query.var.columns:
    query.var["feature_name"] = query.var.index.tolist()
query.var.index = query.var["feature_name"].tolist()

# Use raw counts if available
if query.raw is not None:
    if query.raw.X is not None:
        query.X = query.raw.X

# Preprocessing - critical for celltypist
sc.pp.normalize_total(query, target_sum=1e4)
sc.pp.log1p(query)

# Download model (force_update ensures latest version)
models.download_models(model='Immune_All_Low.pkl', force_update=True)
model = models.Model.load(model='Immune_All_Low.pkl')

# Clean unnecessary data to save memory
del query.uns
del query.obsm
del query.obsp

# Run annotation with majority voting
predictions = celltypist.annotate(query, model='Immune_All_Low.pkl', majority_voting=True)

# Extract predictions
pred_df = pd.DataFrame(predictions.predicted_labels)

# Check results
print(predictions.predicted_labels["predicted_labels"].value_counts())
