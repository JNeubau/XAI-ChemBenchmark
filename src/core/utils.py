from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina


def select_diverse_subset_butina(mol_list, num_to_select, similarity_cutoff=0.65):
    """
    Selects a diverse subset of molecules using Butina clustering.

    Args:
        mol_list (list): A list of RDKit molecule objects.
        num_to_select (int): The desired number of molecules in the diverse subset.
        similarity_cutoff (float): The Tanimoto similarity cutoff for clustering.

    Returns:
        list: A list of RDKit molecule objects representing the diverse subset.
    """
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mol_list]

    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # 3. Cluster the molecules
    cs = Butina.ClusterData(dists, nfps, similarity_cutoff, isDistData=True)

    sorted_clusters = sorted(cs, key=len, reverse=True)

    diverse_subset_indices = []
    for cluster in sorted_clusters:
        if len(diverse_subset_indices) < num_to_select:
            diverse_subset_indices.append(cluster[0])
        else:
            break

    return diverse_subset_indices