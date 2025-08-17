"""
Defines the Markov decision process of generating a molecule.
The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.

"""
import collections
import copy
import itertools
import random
from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.use_deterministic_algorithms(True)


class Result(collections.namedtuple("Result", ["state", "reward", "terminated"])):
    """
    A namedtuple defines the result of a step taken.
    The namedtuple contains the following fields:
        state: The instance reached after taking the action.
        reward: Float. The reward get after taking the action.
        terminated: Boolean. Whether this episode is terminated.
    """

class SortedQueue:
    def __init__(self,
                 num_items,
                 sort_predicate=None
    ):
        self.num_items = num_items
        self.sort_predicate = sort_predicate
        self.data_ = []

    def contains(self, smiles):
        return any(d['id'] == smiles for d in self.data_)

    def insert(self, data):
        assert 'id' in data
        assert 'reward' in data

        if self.contains(data['id']):
            return

        self.data_.append(data)
        self.data_.sort(key=self.sort_predicate, reverse=True)
        self.data_ = self.data_[:self.num_items]

    def extend(self, queue):
        assert isinstance(queue, SortedQueue)

        for data in queue.data_:
            self.insert(data)


class MoleculeEnvironment(ABC):
    """
    Defines the Markov decision process of generating a molecule.
    """

    def __init__(
        self,
        atom_types: np.ndarray,
        init_molecule: str,
        allow_removal: bool = True,
        allow_node_addition: bool = True,
        allow_edge_addition: bool = True,
        allow_no_modification: bool = True,
        allow_bonds_between_rings: bool = True,
        allowed_ring_sizes: list = None,
        target_fn: Any = None,
        max_steps: int = 10,
        record_path: bool = False,
    ):
        self.counter = 0
        self.max_steps = max_steps
        self.target_fn = target_fn

        self.atom_types = atom_types

        self.state = None
        self.init_instance = init_molecule
        atom_types = list(atom_types)
        self.max_new_bonds = dict(
            list(zip(atom_types, self.atom_valences(atom_types))))

        self.allow_removal = allow_removal
        self.allow_node_addition = allow_node_addition
        self.allow_edge_addition = allow_edge_addition
        self.allow_no_modification = allow_no_modification
        self.allow_bonds_between_rings = allow_bonds_between_rings
        self.allowed_ring_sizes = allowed_ring_sizes
        self.valid_actions = set()
        # The status should be 'terminated' if initialize() is not called.
        self.record_path = record_path
        self.path = []
        self.max_bonds = 4
        self.action_counter = 1

    def initialize(self) -> None:
        """Resets the MDP to its initial state."""
        self.state = self.init_instance
        if self.record_path:
            self.path = [self.state]
        self.valid_actions = self.get_valid_actions(force_rebuild=True)
        self.counter = 0
        self.action_counter = 1

    def get_valid_actions(
        self,
        state: Any = None,
        force_rebuild: bool = False,
    ) -> set:
        if state is None:
            if self.valid_actions and not force_rebuild:
                return copy.deepcopy(self.valid_actions)
            state = self.state

        self.valid_actions = self.enumerate_valid_actions(
            state,
            atom_types=self.atom_types,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allowed_ring_sizes=self.allowed_ring_sizes,
            allow_bonds_between_rings=self.allow_bonds_between_rings,
        )
        return copy.deepcopy(self.valid_actions)

    def enumerate_valid_actions(
        self,
        state: str,
        atom_types: np.ndarray,
        allow_removal: bool = True,
        allow_no_modification: bool = True,
        allowed_ring_sizes: list = None,
        allow_bonds_between_rings: bool = True,
    ) -> set:
        if not state:
            return set(copy.deepcopy(atom_types))
        mol = Chem.MolFromSmiles(state)
        if mol is None:
            raise ValueError(f"Received invalid state {state}")

        atom_valences = {
            atom_type: self.atom_valences([atom_type])[0] for atom_type in atom_types
        }
        atoms_with_free_valence = {}

        for i in range(1, max(atom_valences.values())):
            # Only atoms that allow us to replace at least one H with a new bond are
            # # enumerated here
            atoms_with_free_valence[i] = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetNumImplicitHs() >= i
            ]
        valid_actions = set()
        valid_actions.update(
            self._atom_additions(
                mol,
                atom_types=atom_types,
                atom_valences=atom_valences,
                atoms_with_free_valence=atoms_with_free_valence,
            )
        )
        valid_actions.update(
            self._bond_addition(
                mol,
                atoms_with_free_valence=atoms_with_free_valence,
                allowed_ring_sizes=allowed_ring_sizes,
                allow_bonds_between_rings=allow_bonds_between_rings,
            )
        )
        if allow_removal:
            valid_actions.update(self._bond_removal(mol))
        # add the same state
        if allow_no_modification:
            valid_actions.add(state)
        valid_actions = self._validity_double_check(valid_actions)
        return valid_actions

    def _validity_double_check(self, valid_actions: set) -> set:
        """
        Double check the validity of the actions.
        This is a workaround for the fact that some actions may not be valid after
        sanitization.
        """
        valid_actions_checked = set()
        for action in valid_actions:
            mol = Chem.MolFromSmiles(action)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
                valid_actions_checked.add(action)
            except Exception:
                continue
        return valid_actions_checked

    def _atom_additions(
        self,
        state_mol: Chem.Mol,
        atom_types: np.ndarray,
        atom_valences: dict,
        atoms_with_free_valence: dict,
    ) -> set:
        bond_order = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        atom_addition = set()
        for i in bond_order:
            for atom in atoms_with_free_valence[i]:
                for element in atom_types:
                    if atom_valences[element] < i:
                        continue
                    new_state_molecule = Chem.RWMol(state_mol)
                    idx = new_state_molecule.AddAtom(Chem.Atom(element))
                    new_state_molecule.AddBond(atom, idx, bond_order[i])
                    sanitization_result = Chem.SanitizeMol(
                        new_state_molecule, catchErrors=True
                    )
                    # when sanitization fails
                    if sanitization_result:
                        continue
                    atom_addition.add(Chem.MolToSmiles(new_state_molecule))
                    self.action_counter += 1
        return atom_addition

    def _bond_addition(
        self,
        state_mol: Chem.Mol,
        atoms_with_free_valence: dict,
        allowed_ring_sizes: list,
        allow_bonds_between_rings: bool,
    ) -> set:
        bond_orders = [
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ]
        bond_addition = set()
        for valence, atoms in atoms_with_free_valence.items():
            for atom1, atom2 in itertools.combinations(atoms, 2):
                # Get the bond from a copy of the molecule so that SetBondType() doesn't
                # modify the original state.
                bond = Chem.Mol(state_mol).GetBondBetweenAtoms(atom1, atom2)
                new_state_molecule = Chem.RWMol(state_mol)
                # Kekulize the new state to avoid sanitization errors; note that bonds
                # that are aromatic in the original state are not modified (this is
                # enforced by getting the bond from the original state with
                # GetBondBetweenAtoms()).
                Chem.Kekulize(new_state_molecule, clearAromaticFlags=True)
                if bond is not None:
                    if bond.GetBondType() not in bond_orders:
                        continue  # Skip aromatic bonds.
                    # Compute the new bond order as an offset from the current bond order.
                    bond_order = bond_orders.index(bond.GetBondType())
                    bond_order += valence
                    if bond_order < len(bond_orders):
                        idx = bond.GetIdx()
                        bond.SetBondType(bond_orders[bond_order])
                        new_state_molecule.ReplaceBond(idx, bond)
                    else:
                        continue
                # If do not allow new bonds between atoms already in rings.
                elif not allow_bonds_between_rings and (
                    state_mol.molecule.GetAtomWithIdx(atom1).IsInRing()
                    and state_mol.molecule.GetAtomWithIdx(atom2).IsInRing()
                ):
                    continue
                # If the distance between the current two atoms is not in the
                # allowed ring sizes
                elif (
                    allowed_ring_sizes is not None
                    and len(Chem.rdmolops.GetShortestPath(state_mol, atom1, atom2))
                    not in allowed_ring_sizes
                ):
                    continue
                else:
                    new_state_molecule.AddBond(atom1, atom2, bond_orders[valence])
                sanitization_result = Chem.SanitizeMol(
                    new_state_molecule, catchErrors=True
                )
                # When sanitization fails
                if sanitization_result:
                    continue
                bond_addition.add(Chem.MolToSmiles(new_state_molecule))
                self.action_counter += 1

        return bond_addition

    def _bond_removal(
        self, state_mol: Chem.Mol
    ) -> set:
        bond_orders = [
            None,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
        ]
        bond_removal = set()
        for valence in [1, 2, 3]:
            for bond in state_mol.GetBonds():
                # Get the bond from a copy of the molecule so that SetBondType() doesn't
                # modify the original state.
                bond = Chem.Mol(state_mol).GetBondBetweenAtoms(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                )
                if bond is None or bond.GetBondType() not in bond_orders:
                    continue  # Skip aromatic bonds.
                new_state_molecule = Chem.RWMol(state_mol)
                # Kekulize the new state to avoid sanitization errors; note that bonds
                # that are aromatic in the original state are not modified (this is
                # enforced by getting the bond from the original state with
                # GetBondBetweenAtoms()).
                Chem.Kekulize(new_state_molecule, clearAromaticFlags=True)
                # Compute the new bond order as an offset from the current bond order.
                bond_order = bond_orders.index(bond.GetBondType())
                bond_order -= valence
                if bond_order > 0:  # Downgrade this bond.
                    idx = bond.GetIdx()
                    bond.SetBondType(bond_orders[bond_order])
                    new_state_molecule.ReplaceBond(idx, bond)
                    sanitization_result = Chem.SanitizeMol(
                        new_state_molecule, catchErrors=True
                    )
                    # When sanitization fails
                    if sanitization_result:
                        continue

                    bond_removal.add(Chem.MolToSmiles(new_state_molecule))
                    self.action_counter += 1

                elif bond_order == 0:  # Remove this bond entirely.
                    atom1 = bond.GetBeginAtom().GetIdx()
                    atom2 = bond.GetEndAtom().GetIdx()
                    new_state_molecule.RemoveBond(atom1, atom2)
                    sanitization_result = Chem.SanitizeMol(
                        new_state_molecule, catchErrors=True
                    )
                    # When sanitization fails
                    if sanitization_result:
                        continue
                    try:
                        smiles = Chem.MolToSmiles(new_state_molecule)
                    except Exception:
                        continue
                    parts = sorted(smiles.split("."), key=len)
                    # We define the valid bond removing action set as the actions
                    # that remove an existing bond, generating only one independent
                    # molecule, or a molecule and an atom.
                    if len(parts) == 1 or len(parts[0]) == 1:
                        bond_removal.add(parts[-1])
                        self.action_counter += 1
        return bond_removal

    @abstractmethod
    def reward(self):
        pass

    def goal_reached(self) -> bool:
        if not self.target_fn:
            return False
        return self.target_fn(self.state)

    def step(self, action) -> Result:
        if self.counter >= self.max_steps or self.goal_reached():
            raise ValueError("This episode is terminated.")
        if action not in self.valid_actions:
            raise ValueError("Invalid action.")
        self.state = action
        if self.record_path:
            self.path.append(self.state)
        self.valid_actions = self.get_valid_actions(force_rebuild=True)
        self.counter += 1

        result = Result(
            state=self.state,
            reward=self.reward(),
            terminated=(self.counter >= self.max_steps) or self.goal_reached(),
        )
        return result

    def visualize_state(
        self,
        state: str | None = None,
        **kwargs,
    ):
        if state is None:
            state = self.state
        if state is None:
            raise ValueError("No state provided.")
        if isinstance(state, str):
            molecule = Chem.MolFromSmiles(state)
            return Draw.MolToImage(molecule, **kwargs)
        return None

    @staticmethod
    def atom_valences(atom_types):
        periodic_table = Chem.GetPeriodicTable()
        valences = [
            max(list(periodic_table.GetValenceList(atom_type)))
            for atom_type in atom_types
        ]
        return valences