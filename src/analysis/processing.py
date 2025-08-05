import numpy as np


def process_folds(results, smarts_mapping, specific_func):



def process_folds_shap(results, smarts_mapping):
    """
    Process SHAP folds.
    """
    smarts_sorted = {}
    matching_molecules_folds =  {}
    molecules_statistics_folds = {}

    for i, smiles in enumerate(results['smiles']):
        shap_values = results['shap_values'][i]
        feature_names = x_test.columns.tolist()
        expected_values = results['expected_values'][i]
        x_test['smiles'] = smiles

        for m_id, m_shap in enumerate(shap_values):
            m_features = x_test.values[m_id]
            positive_absent = (m_shap >= 0) & (m_features == 0)
            positive_present = (m_shap >= 0) & (m_features > 0)
            negative_absent = (m_shap < 0) & (m_features == 0)
            negative_present = (m_shap < 0) & (m_features > 0)

            presence_increases_prediction = positive_present + negative_absent
            presence_decreases_prediction = negative_present + positive_absent

            abs_m_shap = np.abs(m_shap)
            sorted_shap = np.argsort(abs_m_shap)[::-1]
            sorted_shap = [idx for idx in sorted_shap if abs_m_shap[idx] > 0]
            sorted_shap_feature_names = [feature_names[i] for i in sorted_shap]

            smarts_m = {
                (i, smiles[m_id], feature): smarts_mapping[feature] for feature in sorted_shap_feature_names
            }
            matching_molecules = {s: [] for s in smarts_m.keys()}
            molecules_statistics = {s: {
                    "num_molecules_with_fp": 0,
                    "num_molecules_where_important": 0,
                    "shap_value": 0,
                    "shap_sign": 1,
                    "is_feature_present": False,
                    "capacity_max": 0,
                    "capacity_pred": 0,

                } for s in smarts_m.keys()
            }

