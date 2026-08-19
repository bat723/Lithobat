import re

with open("make_results_dashboard.py", "r") as f:
    code = f.read()

old = """# MSE distribution
bins_mse = np.linspace(0, max(defect_mses.max(), clean_mses.max()) * 1.05, 80)
ax2a.hist(clean_mses, bins=bins_mse, alpha=0.7, color=GREEN, label="Clean", density=True)
ax2a.hist(defect_mses, bins=bins_mse, alpha=0.7, color=RED, label="Defective", density=True)"""

new = """# MSE distribution
if len(clean_mses) > 0 and len(defect_mses) > 0:
    bins_mse = np.linspace(0, max(defect_mses.max(), clean_mses.max()) * 1.05, 80)
elif len(defect_mses) > 0:
    bins_mse = np.linspace(0, defect_mses.max() * 1.05, 80)
else:
    bins_mse = np.linspace(0, 1, 80)
if len(clean_mses) > 0:
    ax2a.hist(clean_mses, bins=bins_mse, alpha=0.7, color=GREEN, label="Clean", density=True)
ax2a.hist(defect_mses, bins=bins_mse, alpha=0.7, color=RED, label="Defective", density=True)"""

code = code.replace(old, new)

old2 = """# Max residual distribution
bins_max = np.linspace(0, 1.05, 80)
ax2b.hist(clean_max_residuals, bins=bins_max, alpha=0.7, color=GREEN,
          label="Clean", density=True)
ax2b.hist(defect_max_residuals, bins=bins_max, alpha=0.7, color=RED,
          label="Defective", density=True)"""

new2 = """# Max residual distribution
bins_max = np.linspace(0, 1.05, 80)
if len(clean_max_residuals) > 0:
    ax2b.hist(clean_max_residuals, bins=bins_max, alpha=0.7, color=GREEN,
              label="Clean", density=True)
if len(defect_max_residuals) > 0:
    ax2b.hist(defect_max_residuals, bins=bins_max, alpha=0.7, color=RED,
              label="Defective", density=True)"""

code = code.replace(old2, new2)

# Also try alternate clean path
code = code.replace(
    'clean_img_dir   = Path("dataset/clean/images")',
    'clean_img_dir   = Path("dataset/clean/images") if Path("dataset/clean/images").exists() else Path("dataset/clean")'
)

with open("make_results_dashboard.py", "w") as f:
    f.write(code)

print("Patched!")
