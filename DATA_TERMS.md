# Data terms

[`LICENSE`](LICENSE) (MIT) covers **the code in this repository and nothing
else**. The dataset has separate, upstream terms, recorded here.

This file was split out of `LICENSE` so that automated licence detection reads
the code licence correctly. No wording was weakened in the split.

## What the dataset is, and that it is not redistributed here

The project uses the MoleculeNet **"Lipophilicity"** file: experimental
logD<sub>7.4</sub> values from an AstraZeneca deposition in ChEMBL.

It is **not redistributed in this repository.** `data/raw/` is git-ignored.
[`src/admet/download.py`](src/admet/download.py) fetches it from its public
source at build time and pins the exact bytes by SHA-256
(`aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`,
282,606 bytes), so a run is reproducible without anyone re-hosting the file.

## ChEMBL licence

ChEMBL data is released under the **Creative Commons
Attribution-ShareAlike 3.0 Unported** licence. Stated under "Current Release"
at <https://www.ebi.ac.uk/chembl/>, linking to
<https://creativecommons.org/licenses/by-sa/3.0/>.

## EMBL-EBI terms of use

From <https://www.ebi.ac.uk/about/terms-of-use> (Data, section 3), verbatim:

> EMBL-EBI itself places no additional restrictions on the use or
> redistribution of the data … other than those provided by the original data
> owners, unless otherwise specified in these Terms of Use.

Section 4 adds that the original data "may be subject to rights claimed by
third parties". General item 2 requires attribution:

> EMBL-EBI expects attribution (e.g. in publications, services or products)
> for the use of any of its Data Resources and Tools in accordance with good
> scientific practice.

## Host of the file

DeepChem distributes the file and licenses **its own code** under MIT
(<https://github.com/deepchem/deepchem/blob/master/LICENSE> — the file reads
"Copyright 2017 PandeLab" followed by the standard MIT permission text).

The DeepChem documentation describes the dataset verbatim as:

> LIPO | Experimental results of octanol/water distribution coefficient
> (logD at pH 7.4) | Molecules | 4 200

(<https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html>,
which links that row to <https://arxiv.org/abs/1703.00564>.)

## Two things this file deliberately does not assert

Both were checked against primary sources and **could not be verified**, so
they are not claimed:

1. **The exact ChEMBL deposition identifier.** Working notes in this
   repository have cited `CHEMBL3301363`, and a secondary summary gave DOI
   `10.6019/chembl3301361`. Neither was confirmed against a primary source in
   this work, so `download.py` describes the provenance without asserting an
   identifier.
2. **A licence for the data *file* as distinct from the code that distributes
   it.** The MoleculeNet datasets page returns HTTP 404, and neither host
   declares a licence for the file itself.

Anyone **redistributing** this data should establish both points for
themselves rather than rely on this note. Downloading it from the public
source, as this project does, does not depend on either.

## Attribution to use

> Experimental logD<sub>7.4</sub> data from ChEMBL (AstraZeneca deposited data
> set), distributed as the MoleculeNet "Lipophilicity" benchmark. ChEMBL data
> is made available under CC BY-SA 3.0.

ChEMBL asks to be cited as: Zdrazil B, Felix E, Hunter F, et al. "The ChEMBL
Database in 2023: a drug discovery platform spanning multiple bioactivity data
types and time periods." *Nucleic Acids Res.* 2023: gkad1004.
doi:[10.1093/nar/gkad1004](https://doi.org/10.1093/nar/gkad1004)

Note that CC BY-**SA** is a share-alike licence: a derivative *of the data*
carries obligations that MIT does not. That is one more reason the data is kept
out of this repository rather than vendored into it.
