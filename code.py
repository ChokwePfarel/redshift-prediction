import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr 

base_dir = 'C:/Users/chokw/OneDrive/Desktop/researchProject/ml_esac'
path = os.path.join(base_dir, 'spectra.npy')
metadata_path = os.path.join(base_dir, 'metadata.npy') # Using 'metadata.npy' as per your last file
test_size = 0.033
random_seed = 44
n_components = 2  # Number of PCA components

# --- Data Loading and Redshift Extraction ---
try:
    F = np.load(path)
    M = np.load(metadata_path, allow_pickle=True)
    print(M, type(M), M.dtype, M.shape)

    # Simplified and corrected logic for extracting redshift (redshift is the 1st column, index 0)
    try:
        if M.ndim > 1:
            # If M is 2D or higher, select the first column (index 0)
            redshift = M[:, 0]
        elif M.dtype.names is not None and 'z' in M.dtype.names:
             # If M is a structured array, use the 'z' field
             redshift = M['z']
        elif M.ndim == 1:
            # If M is 1D, assume it is the redshift array itself
            redshift = M
        else:
            print("Warning:Redshift extraction failed. Using random values for demonstration.")
            redshift = np.random.rand(F.shape[0]) * 3

    except Exception:
        print("Warning: Redshift extraction failed. Using random values for demonstration.")
        redshift = np.random.rand(F.shape[0]) * 3

except FileNotFoundError as e:
    print(f"Error: Required file not found. Check paths. {e}")
    F = np.random.rand(200, 4001)
    redshift = np.random.rand(200) * 3

# Ensuring F and redshift match size after reduction
#Keep half of the spectra

F = F[:len(F) // 2, :]
redshift = redshift[:len(redshift) // 2]

# Wavelength array (only for plotting, not ML)
num_flux_points = F.shape[1]
wavs = np.linspace(4000, 8000, num_flux_points)

# --- PCA Feature Extraction --------------------
pca = PCA(n_components=n_components)
F_pca = pca.fit_transform(F)

# Print PCA info--------------------------------
print(f"Shape of flux data: {F.shape}")
print(f"Shape of PCA-transformed flux: {F_pca.shape}")
print(f"Explained variance ratio of first {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")
print("--------------------------------------------------")

# --- Model Training and Prediction ---
X_train, X_test, y_train, y_test = train_test_split(
    F_pca, redshift, test_size=test_size, random_state=random_seed
)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=random_seed, n_jobs=-1)
rf_regressor.fit(X_train, y_train)

# Predict redshift for the test set and all data
y_pred_test = rf_regressor.predict(X_test)
z_predicted_all = rf_regressor.predict(F_pca)

# --- Performance Evaluation --------

print(f"Random Forest Performance on Test Set:")
print(f"  Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"  R-squared ($R^2$): {r2_score(y_test, y_pred_test):.4f}")
print("")

# Calculate the Pearson correlation coefficient (r) on the test set
r, p_value = pearsonr(y_test, y_pred_test)
print(f"Correlation Coefficient (r) on Test Set: {r:.4f}")
print("--------------------------------------------------")

# --- Visualization (Spectra Plot - Remains the same) ----------

cmap = plt.cm.viridis
norm = Normalize(vmin=np.min(z_predicted_all), vmax=np.max(z_predicted_all))

plt.figure(figsize=(12, 6))
for i in range(F.shape[0]):
    color = cmap(norm(z_predicted_all[i]))
    plt.plot(wavs, F[i, :], color=color, alpha=0.5, linewidth=0.5)

plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Flux (Arbitrary Units)')
plt.title(f'Spectra Colored by Predicted Redshift ($\hat{{z}}$ from Random Forest, $N_{{PCA}}={n_components}$)')

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Predicted Redshift ($\hat{{z}}$)')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# --- Visualization (PCA Scatter Plot - Remains the same) ----------------
plt.figure(figsize=(8, 6))
plt.scatter(F_pca[:, 0], F_pca[:, 1], c=z_predicted_all, cmap=cmap, norm=norm, s=10, alpha=0.7)

pc1_variance = pca.explained_variance_ratio_[0]*100
pc2_variance = pca.explained_variance_ratio_[1]*100
plt.xlabel(f'PCA Component 1 ({pc1_variance:.1f}%)')
plt.ylabel(f'PCA Component 2 ({pc2_variance:.1f}%)')
plt.title(f'PCA Feature Space Colored by Predicted Redshift ($\hat{{z}}$)')

sm_pca = ScalarMappable(cmap=cmap, norm=norm)
sm_pca.set_array([])
cbar_pca = plt.colorbar(sm_pca, ax=plt.gca())
cbar_pca.set_label('Predicted Redshift ($\hat{{z}}$)')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()



#-----------------------------------------------------

from sklearn.svm import SVR

# Initialize and train the SVR model
svr_regressor = SVR(kernel='rbf', C=10, gamma='scale')
svr_regressor.fit(X_train, y_train)

# Predict redshift for the test set and all data
y_pred_test_svr = svr_regressor.predict(X_test)
z_predicted_all_svr = svr_regressor.predict(F_pca)

# --- Performance Evaluation for SVR ---
print("\nSupport Vector Regression (SVR) Performance on Test Set:")
print(f"  Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_test_svr):.4f}")
print(f"  R-squared ($R^2$): {r2_score(y_test, y_pred_test_svr):.4f}")

# Calculate the Pearson correlation coefficient (r) for SVR
r_svr, p_value_svr = pearsonr(y_test, y_pred_test_svr)
print(f"  Correlation Coefficient (r): {r_svr:.4f}")
print("--------------------------------------------------")

# ---Visualization: PCA Scatter (SVR Predicted Redshift) ------------------

plt.figure(figsize=(8, 6))
plt.scatter(F_pca[:, 0], F_pca[:, 1], c=z_predicted_all_svr, cmap=cmap, norm=norm, s=10, alpha=0.7)

plt.xlabel(f'PCA Component 1 ({pc1_variance:.1f}%)')
plt.ylabel(f'PCA Component 2 ({pc2_variance:.1f}%)')
plt.title(f'PCA Feature Space Colored by Predicted Redshift (SVR, $N_{{PCA}}={n_components}$)')

sm_pca_svr = ScalarMappable(cmap=cmap, norm=norm)
sm_pca_svr.set_array([])
cbar_pca_svr = plt.colorbar(sm_pca_svr, ax=plt.gca())
cbar_pca_svr.set_label('Predicted Redshift ($\hat{{z}}$)')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
