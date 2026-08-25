import subprocess
import sys

scripts = [
    "a1_uav_diagram.py",
    "a2_nonlinear_calibration.py",
    "a3_resolution_leakage.py",
    "a4_aliasing.py",
    "a5_first_order_ode.py",
    "a6_first_order_filter.py",
    "a7_second_order_id.py",
    "a8_combined_system.py",
]

for s in scripts:
    print(f"\n===== running {s} =====")
    subprocess.run([sys.executable, s], check=True)

print("\nAll done. Figures are in ./figures_output/")
