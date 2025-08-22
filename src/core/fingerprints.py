"""Functions for generating fingerprints."""
from functools import partial
from typing import Callable, Literal

import numpy as np
import pandas as pd
import skfp
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from skfp.fingerprints import (
    ECFPFingerprint,
    EStateFingerprint,
    FunctionalGroupsFingerprint,
    LayeredFingerprint,
    MACCSFingerprint,
    PatternFingerprint,
    TopologicalTorsionFingerprint,
)


class Fingerprints:
    """
    Class for generating fingerprints.
    """

    _fingerprints: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Register a fingerprint function with a given name.

        This method is used as a decorator to register a fingerprint function
        under a specified name. The registered function can later be retrieved
        and used to generate fingerprints.

        :param name: The name to register the fingerprint function under.
        :return: A decorator that registers the fingerprint function.
        """

        def decorator(func: Callable) -> Callable:
            cls._fingerprints[name] = func
            return func

        return decorator

    def apply(self, names: list, smiles: list, **kwargs) -> tuple:
        """
        Apply the fingerprint function with the given name.
        :param name: name of the fingerprint function.
        :param smiles: list of SMILES strings.
        :param kwargs: additional arguments for the fingerprint function.
        :return: list of fingerprints and feature names.
        """
        fingerprints = []
        features_names = []
        for name in names:
            if name not in self._fingerprints:
                raise ValueError(f"Fingerprint function '{name}' is not defined.")
            fp = self._fingerprints[name](**kwargs[name])
            fps = fp.fit_transform(smiles)
            if name != 'descriptor':
                fps_names = [f'{name}_feature_{i}' for i in range(fps.shape[1])]
            else:
                fps_names = fp.get_feature_names_out()
            # fps_names = [f'{name}_feature_{i}' for i in range(fps.shape[1])]
            fingerprints.append(fps)
            features_names.extend(fps_names)
        fingerprints = np.concat(fingerprints, axis=1)
        return fingerprints, features_names


@Fingerprints.register("ecfp")
def ecfp_fingerprint(size: int = 1024, radius: int = 2, count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate ECFP fingerprints.
    :param size: size of the fingerprint.
    :param radius: radius of neighbors to consider.
    :param count: whether to include counts or to use binary values.
    :return: ECFP fingerprint.
    """
    return ECFPFingerprint(fp_size=size, radius=radius, include_chirality=True, count=count)


@Fingerprints.register("maccs")
def maccs_fingerprint(count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate MACCS fingerprints.
    :param count: whether to include counts or to use binary values.
    :return: Maccs fingerprint.
    """
    return MACCSFingerprint(count=count)


@Fingerprints.register("functional_groups")
def functional_groups_fingerprint(count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate functional groups fingerprints.
    :param count: whether to include counts or to use binary values.
    :return: functional groups fingerprint.
    """
    return FunctionalGroupsFingerprint(count=count)


@Fingerprints.register("estate")
def estate_fingerprint(variant: Literal["bit", "count", "sum"] = "sum") -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate estate fingerprints.
    :param variant: type of estate fingerprint.
    :return: estate fingerprint.
    """
    return EStateFingerprint(variant=variant)


@Fingerprints.register("layered")
def layered_fingerprint(size: int = 1024, linear_paths_only: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate layered fingerprints.
    :param size: size of the fingerprint.
    :param linear_paths_only: whether to consider only linear paths.
    :return: layered fingerprint.
    """
    return LayeredFingerprint(fp_size=size, linear_paths_only=linear_paths_only)


@Fingerprints.register("pattern")
def pattern_fingerprint(size: int = 1024, tautomers: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate pattern fingerprints.
    :param size: size of the fingerprint.
    :param tautomers: whether to consider tautomers.
    :return: pattern fingerprint.
    """
    return PatternFingerprint(fp_size=size, tautomers=tautomers)


@Fingerprints.register("topological")
def topological_fingerprint(size: int = 1024, torsion_atoms: int = 4, count: bool = False) -> skfp.bases.BaseFingerprintTransformer:
    """
    Generate topological fingerprints.
    :param size: size of the fingerprint.
    :param torsion_atoms: how many atoms to consider in torsion.
    :param count: whether to include counts or to use binary values.
    :return: topological torsion fingerprint.
    """
    return TopologicalTorsionFingerprint(fp_size=size, torsion_atom_count=torsion_atoms, include_chirality=True, count=count)

@Fingerprints.register("descriptor")
def descriptor() -> object:
    return CustomDescriptor()

class CustomDescriptor:
    def __init__(self):
        self.desc = {
            'radius': skfp.descriptors.radius,
            'diameter': skfp.descriptors.diameter,
            "num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
            "num_h_acceptors": rdMolDescriptors.CalcNumLipinskiHBA,
            "num_h_donors": rdMolDescriptors.CalcNumLipinskiHBD,
            "tpsa": rdMolDescriptors.CalcTPSA,
            "mol_wt": rdMolDescriptors.CalcExactMolWt,
            "o%": partial(self.calculate_percentage, idx=8),
            "n%": partial(self.calculate_percentage, idx=7),
            "c%": partial(self.calculate_percentage, idx=6),
        }

    def calculate_percentage(self, mol: Chem.Mol, idx: int) -> float:
        """
        Calculate the percentage of oxygen atoms in the molecule.
        :param mol: RDKit molecule object.
        :return: percentage of oxygen atoms.
        """
        num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == idx)
        num_atoms = mol.GetNumAtoms()
        return num_oxygen / num_atoms

    def fit_transform(self, smiles: list) -> np.ndarray:
        """
        Custom descriptor for fingerprints.
        :param smiles: list of SMILES strings.
        :return: array with custom descriptors.
        """
        descriptors = np.zeros((len(smiles), len(self.desc.keys())))
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            for j, (name, func) in enumerate(self.desc.items()):
                descriptors[i, j] = func(mol)
        return descriptors

    def get_feature_names_out(self) -> list:
        """
        Get feature names for custom descriptors.
        :return: list of feature names.
        """
        return list(self.desc.keys())

def fingerprints_dataset(
    df: pd.DataFrame, smiles_col: str, target_col: str, fingerprint_type: Literal["ecfp", "maccs", "functional_groups", "estate", "layered", "pattern", "rdf", "topological"], **kwargs
) -> pd.DataFrame:
    """
    Generates fingerprints for the given dataset.
    :param df: dataframe with SMILES and target columns.
    :param smiles_col: name of the column with SMILES.
    :param target_col: name of the target column.
    :param fingerprint_type: type of fingerprint to generate.
    :param kwargs: additional arguments for the fingerprint generation.
    :return: dataframe with generated fingerprints.
    """
    smiles_list = df[smiles_col].tolist()
    target_list = df[target_col].tolist()
    feature_list, features_names = Fingerprints().apply(fingerprint_type, smiles_list, **kwargs["kwargs"])
    df_features = pd.DataFrame(feature_list, columns=features_names)
    df_features[target_col] = target_list
    df_features[smiles_col] = smiles_list
    return df_features

def smiles_to_fingerprint(smiles: str, fingerprint_type: str, **kwargs) -> tuple:
    """
    Convert SMILES to fingerprint.
    :param smiles: SMILES string.
    :param fingerprint_type: type of fingerprint to generate.
    :param kwargs: additional arguments for the fingerprint generation.
    :return: tuple with fingerprint and feature names.
    """
    fp, feature_names = Fingerprints().apply(fingerprint_type, [smiles], **kwargs)
    return fp, feature_names

if __name__ == "__main__":
    df = pd.read_csv("./data/new_maccs_merged.csv")
    smiles = df["smiles"].tolist()[0]
    fp, f_names = smiles_to_fingerprint(smiles, "maccs", count=False)
    common_columns = df.columns.intersection(f_names)
    common_indices = [i for i, name in enumerate(f_names) if name in common_columns]
    filtered_fp = [fp[0][i] for i in common_indices]
    print(filtered_fp, f_names[common_indices])

    # df_fingerprints = fingerprints_dataset(df, "smiles", "capacity_max", "maccs", kwargs={"count": True})
    # print(df_fingerprints.head())
    # maccs = MACCSFingerprint(count=False)
    # print(maccs.get_feature_names_out())
    # df_fingerprints.to_csv("../../../data/fingerprints_maccs/data_experts1.csv", index=False)
