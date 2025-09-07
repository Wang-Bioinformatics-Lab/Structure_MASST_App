import urllib.parse
import json
import re
import math
from urllib.parse import quote_plus

def _safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def build_dashboard_eic_url(usi: str, xic_mz, xic_tolerance, xic_rt_window=None) -> str:
    """
    Robust GNPS2 dashboard URL builder.
    - usi: string ending with ':scan:<number>'
    - xic_mz: numeric or string; if not numeric, left empty in URL and no m/z zoom applied
    - xic_tolerance: numeric or string (passed through)
    - xic_rt_window: numeric or string; omitted if not numeric
    """
    # Extract scan
    m = re.search(r":scan:(\d+)$", str(usi))
    ms2_identifier = f"MS2:{m.group(1)}" if m else "None"

    # Coerce numerics safely
    xmz = _safe_float(xic_mz)
    rt  = _safe_float(xic_rt_window)

    # Base query params
    params = {
        "xic_mz": f"{xmz:.6f}" if xmz is not None else "",
        "xic_formula": "",
        "xic_peptide": "",
        "xic_tolerance": xic_tolerance,
        "xic_ppm_tolerance": 10,
        "xic_tolerance_unit": "Da",
        "xic_rt_window": f"{rt:.8f}" if rt is not None else "",
        "xic_norm": "False",
        "xic_file_grouping": "MZ",
        "xic_integration_type": "AUC",
        "show_ms2_markers": "True",
        "ms2marker_color": "blue",
        "ms2marker_size": 5,
        "ms2_identifier": ms2_identifier,
        "show_lcms_2nd_map": "False",
        # map_plot_zoom added below only if we can compute it
        "polarity_filtering": "None",
        "polarity_filtering2": "None",
        "tic_option": "TIC",
        "overlay_usi": "None",
        "overlay_mz": "None",
        "overlay_rt": "None",
        "overlay_color": "",
        "overlay_size": "",
        "overlay_hover": "",
        "overlay_filter_column": "",
        "overlay_filter_value": "",
        "feature_finding_type": "Off",
        "feature_finding_ppm": 10,
        "feature_finding_noise": 10000,
        "feature_finding_min_peak_rt": 0.05,
        "feature_finding_max_peak_rt": 1.5,
        "feature_finding_rt_tolerance": 0.3,
        "massql_statement": "QUERY scaninfo(MS2DATA)",
        "sychronization_session_id": "cd9f3c5172c64c55b2c3efd6d2d49b73",
        "chromatogram_options": "[]",
        "comment": "",
        "map_plot_color_scale": "Hot_r",
        "map_plot_quantization_level": "Medium",
        "plot_theme": "plotly_white",
    }

    # Only set m/z zoom if xic_mz is numeric
    if xmz is not None:
        y_min, y_max = xmz - 3.0, xmz + 3.0
        map_zoom = {
            "xaxis.range[0]": 0,
            "xaxis.range[1]": 1,
            "yaxis.range[0]": y_min,
            "yaxis.range[1]": y_max,
        }
        params["map_plot_zoom"] = urllib.parse.quote(json.dumps(map_zoom))

    # Build the base URL
    base = "https://dashboard.gnps2.org//?"
    query = urllib.parse.urlencode(params, doseq=False)

    # Fragment (USI state)
    fragment_obj = {"usi": usi, "usi_select": usi, "usi2": ""}
    fragment = urllib.parse.quote(json.dumps(fragment_obj))

    return base + query + "#" + fragment


# build links for best spectral match and modification site
def build_spectraresolver_link(usi1: str, usi2: str) -> str:
    """
    Build a spectra-resolver comparison link from two USIs.
    
    Parameters
    ----------
    usi1 : str
        First USI (query spectrum)
    usi2 : str
        Second USI (library spectrum)
    """
    usi1_q = quote_plus(usi1)
    usi2_q = quote_plus(usi2)
    
    return (
        f"http://metabolomics-usi.gnps2.org/dashinterface"
        f"?usi1={usi1_q}"
        f"&usi2={usi2_q}"
        f"&width=10.0&height=6.0&mz_min=None&mz_max=None"
        f"&max_intensity=125&annotate_precision=4&annotation_rotation=90"
        f"&cosine=standard&fragment_mz_tolerance=0.05"
        f"&grid=True&annotate_peaks=%5B%5B%5D%2C%20%5B%5D%5D"
    )

if __name__ == "__main__":

    # Example usage
    usi_example = "mzspec:MSV000079949:ccms_peak/3D_Mouse_GF_SPF/SPF1/SPF1_Ile1_BD4_01_21892.mzXML:scan:2449"
    print(build_dashboard_eic_url(usi_example, xic_mz=556.3622, xic_tolerance=0.5))
