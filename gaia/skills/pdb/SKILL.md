---
name: pdb
description: Use this skill whenever the user wants to read, parse, analyze, or manipulate PDB (Protein Data Bank) files. This includes extracting atomic coordinates, analyzing protein structures, identifying residues, chains, ligands, computing distances between atoms, visualizing molecular structures, and converting between molecular file formats. If the user mentions a .pdb file or asks about protein/molecular structure data, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# PDB (Protein Data Bank) File Processing Guide

## Overview

This guide covers reading, parsing, and analyzing PDB files using Python libraries. PDB files store 3D structural data for biological macromolecules (proteins, nucleic acids) and are the standard format in structural biology. The primary library is `BioPython` with its `Bio.PDB` module.

## Quick Start

```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

# Iterate through the hierarchy: Structure -> Model -> Chain -> Residue -> Atom
for model in structure:
    for chain in model:
        print(f"Chain {chain.id}: {len(list(chain.get_residues()))} residues")
```

## Dependencies

```bash
pip install biopython pandas numpy
```

## PDB File Format Basics

PDB files are fixed-width text files. Key record types:

| Record | Description |
|--------|-------------|
| HEADER | File metadata |
| ATOM | Standard residue atom coordinates |
| HETATM | Non-standard residue (ligands, water) |
| CONECT | Bond connectivity |
| SEQRES | Full sequence of residues |
| HELIX/SHEET | Secondary structure annotations |
| REMARK | Comments and metadata |
| END | End of file |

### ATOM Record Format
```
ATOM      1  N   ALA A   1      27.360  24.430  10.100  1.00 20.00           N
Columns:  1-6   Record name
          7-11  Atom serial number
         13-16  Atom name
         17     Alternate location
         18-20  Residue name
         22     Chain ID
         23-26  Residue sequence number
         31-38  X coordinate (Angstroms)
         39-46  Y coordinate
         47-54  Z coordinate
         55-60  Occupancy
         61-66  B-factor (temperature factor)
         77-78  Element symbol
```

## Python Libraries

### BioPython (Bio.PDB) - Full-Featured PDB Parser

#### Parse a PDB File
```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("my_protein", "structure.pdb")

# Access hierarchy
model = structure[0]  # First model
chain = model["A"]    # Chain A
residue = chain[(" ", 100, " ")]  # Residue 100

# Get atom coordinates
for atom in residue:
    print(f"{atom.get_name()}: {atom.get_vector()}")
```

#### List All Chains and Residues
```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

for model in structure:
    for chain in model:
        residues = [r for r in chain.get_residues() if r.id[0] == " "]  # Standard residues only
        print(f"Chain {chain.id}: {len(residues)} residues")
        for res in residues[:5]:
            print(f"  {res.get_resname()} {res.id[1]}")
```

#### Extract Atom Coordinates
```python
from Bio.PDB import PDBParser
import numpy as np

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

# Get all CA (alpha carbon) atoms
ca_atoms = []
for model in structure:
    for chain in model:
        for residue in chain:
            if "CA" in residue:
                atom = residue["CA"]
                ca_atoms.append({
                    "chain": chain.id,
                    "residue": residue.get_resname(),
                    "resid": residue.id[1],
                    "coords": atom.get_vector().get_array()
                })

print(f"Found {len(ca_atoms)} CA atoms")
```

#### Compute Distance Between Atoms
```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

model = structure[0]
chain = model["A"]

# Distance between two residues' CA atoms
atom1 = chain[(" ", 10, " ")]["CA"]
atom2 = chain[(" ", 50, " ")]["CA"]

distance = atom1 - atom2  # BioPython overloads '-' for distance
print(f"Distance: {distance:.2f} Angstroms")
```

#### Extract Ligands (HETATM Records)
```python
from Bio.PDB import PDBParser

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] not in (" ", "W"):  # Not standard residue or water
                print(f"Ligand: {residue.get_resname()} Chain {chain.id} #{residue.id[1]}")
                for atom in residue:
                    print(f"  {atom.get_name()}: {atom.get_vector()}")
```

#### Extract Sequence from Structure
```python
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

ppb = PPBuilder()
for pp in ppb.build_peptides(structure):
    print(f"Sequence: {pp.get_sequence()}")
```

### Manual Parsing - Lightweight Alternative

#### Parse ATOM Records Directly
```python
def parse_pdb(filepath):
    """Parse ATOM/HETATM records from a PDB file."""
    atoms = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atoms.append({
                    "record": line[0:6].strip(),
                    "serial": int(line[6:11].strip()),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resid": int(line[22:26].strip()),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                    "occupancy": float(line[54:60]) if line[54:60].strip() else 1.0,
                    "bfactor": float(line[60:66]) if line[60:66].strip() else 0.0,
                    "element": line[76:78].strip() if len(line) > 76 else "",
                })
    return atoms

atoms = parse_pdb("structure.pdb")
print(f"Total atoms: {len(atoms)}")
```

#### Convert PDB to DataFrame
```python
import pandas as pd

atoms = parse_pdb("structure.pdb")
df = pd.DataFrame(atoms)

# Analyze
print(df.groupby("chain")["resid"].nunique())  # Residues per chain
print(df[df["record"] == "HETATM"]["resname"].unique())  # Ligand types
```

## Common Tasks

### Extract Specific Chain
```python
from Bio.PDB import PDBParser, PDBIO, Select

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

io = PDBIO()
io.set_structure(structure)
io.save("chain_A.pdb", ChainSelect("A"))
```

### Get Secondary Structure Info
```python
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")
model = structure[0]

dssp = DSSP(model, "structure.pdb")
for key in dssp.keys():
    chain, resid = key
    ss = dssp[key][2]  # Secondary structure assignment
    print(f"Chain {chain} Res {resid[1]}: {ss}")
```

### Count Residues by Type
```python
from Bio.PDB import PDBParser
from collections import Counter

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "structure.pdb")

residue_counts = Counter()
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":
                residue_counts[residue.get_resname()] += 1

for resname, count in residue_counts.most_common():
    print(f"{resname}: {count}")
```

## Quick Reference

| Task | Best Tool | Example |
|------|-----------|---------|
| Parse PDB | BioPython | `PDBParser().get_structure(id, file)` |
| Get coordinates | Bio.PDB | `atom.get_vector()` |
| Atom distances | Bio.PDB | `atom1 - atom2` |
| Extract sequence | Bio.PDB | `PPBuilder().build_peptides(struct)` |
| Find ligands | Bio.PDB | Check `residue.id[0]` |
| Extract chain | PDBIO + Select | `io.save("out.pdb", ChainSelect("A"))` |
| Manual parse | Python | Parse fixed-width ATOM lines |
| To DataFrame | pandas | `pd.DataFrame(parse_pdb(file))` |
