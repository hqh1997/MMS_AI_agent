import numpy as np
import sys
sys.path.append("./")
from utils.geodis_toolkits_geo_revise import randompoint, geodismap
import pandas as pd
import os
import SimpleITK as sitk
from scipy.ndimage import binary_closing


seg_path = '/path/to/seg'
prob_path = '/path/to/prob'
gd_path = '/path/to/gd'
img_path = '/path/to/imagesTr'
save_dir = '/path/to/refined'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


SPREAD_FACTOR = 1.0 

def paper_fusion_formula(Pf, Ef_raw, Eb_raw, gamma=1.0):
    """
   
    Args:
        Pf:  P_f
        Ef_raw: geodismap (e^-D_f)
        Eb_raw: geodismap  (e^-D_b)
        gamma: ( Sigma)
    """
    epsilon = 1e-10
    if gamma != 1.0:
        V_f = np.power(Ef_raw, 1.0 / gamma)
        V_b = np.power(Eb_raw, 1.0 / gamma)
    else:
        V_f = Ef_raw
        V_b = Eb_raw
    denominator = V_f + V_b + epsilon
    E_f = V_f / denominator

    # alpha_i = e^-min(D_f, D_b) = max(e^-D_f, e^-D_b) = max(V_f, V_b)
    alpha = np.maximum(V_f, V_b)

    # R_i^f = (1 - alpha) * P_f + alpha * E_f
    R_f = (1 - alpha) * Pf + alpha * E_f
    
    return R_f, alpha

seg_files = sorted(os.listdir(seg_path))
click_records = []

print(f"Starting refinement (Exact Paper Implementation)...")

for name in seg_files:
    print(f"Processing case: {name}")


    seg_sitk = sitk.ReadImage(os.path.join(seg_path, name))
    seg_arr = sitk.GetArrayFromImage(seg_sitk)

    prob_name = name.replace('_rnet.nii.gz', '_rnet_prob.nii.gz')
    prob_file_path = os.path.join(prob_path, prob_name)
    if not os.path.exists(prob_file_path):
        print(f"  - Skipping: Probability map not found.")
        continue

    Pf = sitk.GetArrayFromImage(sitk.ReadImage(prob_file_path)).astype(np.float32)
    if Pf.max() > 1.05:
        Pf = Pf / 255.0
    Pf = np.clip(Pf, 0, 1)
    
    gd_arr = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(gd_path, name.replace('org_rnet.nii.gz', 'seg.nii.gz'))))
    img_arr = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(img_path, name.replace('_rnet.nii.gz', '.nii.gz'))))
    img_arr_processed = np.expand_dims(img_arr.astype(np.float32), axis=0)

    over_seg = np.where(seg_arr - gd_arr == 1, 1, 0)
    under_seg = np.where(seg_arr - gd_arr == -1, 1, 0)

    sb_refine, sb_clicks = randompoint(over_seg, mode='smart')
    sf_refine, sf_clicks = randompoint(under_seg, mode='smart')
    
    click_records.append({'id': name, 'sb_clicks': sb_clicks, 'sf_clicks': sf_clicks})

    if sb_clicks == 0 and sf_clicks == 0:
        print(f"  - No refinement needed.")
        refined_seg_arr = seg_arr
    else:
        print(f"  - Clicks: BG={sb_clicks}, FG={sf_clicks}")


        Ef_raw, Eb_raw = geodismap(sf_refine, sb_refine, img_arr_processed)

        R_f, alpha = paper_fusion_formula(Pf, Ef_raw, Eb_raw, gamma=SPREAD_FACTOR)

        threshold = 0.5
        refined_seg_arr = (R_f > threshold).astype(np.uint8)

        refined_seg_arr = binary_closing(refined_seg_arr, structure=np.ones((3,3,3))).astype(np.uint8)

        if sf_clicks > 0:
            refined_seg_arr[sf_refine > 0] = 1
        if sb_clicks > 0:
            refined_seg_arr[sb_refine > 0] = 0


        if seg_arr.sum() > 0 and refined_seg_arr.sum() == 0:
            print(f"  - Warning: Result empty, reverting.")
            refined_seg_arr = seg_arr.copy()

    refined_sitk = sitk.GetImageFromArray(refined_seg_arr)
    refined_sitk.CopyInformation(seg_sitk)
    sitk.WriteImage(refined_sitk, os.path.join(save_dir, name.replace('_rnet.nii.gz', '_rnet_refined.nii.gz')))
    print(f"  - Saved.")


pd.DataFrame(click_records).to_csv(os.path.join(save_dir, 'refinement_clicks_log.csv'), index=False)
print("\nDone.")
