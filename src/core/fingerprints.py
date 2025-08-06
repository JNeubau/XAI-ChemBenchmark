"""Functions for generating fingerprints."""
from typing import Callable, Literal

import pandas as pd
import skfp
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

    def apply(self, name, smiles, **kwargs) -> tuple:
        """
        Apply the fingerprint function with the given name.
        :param name: name of the fingerprint function.
        :param smiles: list of SMILES strings.
        :param kwargs: additional arguments for the fingerprint function.
        :return: list of fingerprints and feature names.
        """
        if name in self._fingerprints:
            fp = self._fingerprints[name](**kwargs)
            fingerprints = fp.fit_transform(smiles)
            #features_names = fp.get_feature_names_out()
            features_names = [f'feature_{i}' for i in range(fingerprints.shape[1])]
            return fingerprints, features_names
        raise ValueError(f"Fingerprint function '{name}' is not defined.")


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
