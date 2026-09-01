# Data Licensing and Attribution

The **source code** in this repository is licensed under the MIT License (see
[`LICENSE`](LICENSE)).

The **connectome datasets are not bundled in this repository**. They were
produced and published by other research groups and must be obtained from their
original sources. This repository does not relicense them. Each dataset remains
under the terms of its original publication and source. The MIT License
covering the code does not extend to downloaded or derived data files.

If you reuse any connectome data, you must comply with the license of its
original source and cite the original publication, not just this repository.

## Per-dataset terms

| Dataset                                                     | Original source / publication                                                                 | License of the original data |
|------------------------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------|
| *C. elegans* hermaphrodite                                  | White et al. 1986, *Phil. Trans. R. Soc. B*; Cook et al. 2019, *Nature* (WormWiring)          | Under the terms of the original publications / WormWiring; consult the source before reuse |
| *C. elegans* male                                           | Cook et al. 2019, *Nature* (WormWiring)                                                       | Under the terms of the original publication / WormWiring; consult the source before reuse |
| *Ciona intestinalis*                                       | Ryan et al. 2016, *eLife*, doi:10.7554/eLife.16962                                            | eLife article content is CC BY 4.0; attribute Ryan et al. 2016 |
| *Drosophila* larva                                          | Winding et al. 2023, *Science*                                                               | Under the terms of the original *Science* publication and its supplementary data; consult the source before reuse |
| Adult *Drosophila* regions                                  | FlyWire - Dorkenwald et al. 2024, *Nature*; Schlegel et al. 2024 (FlyWire connectome) | FlyWire connectome data are released under CC BY-NC 4.0 (non-commercial, attribution); see the FlyWire data-sharing terms at https://flywire.ai and codex.flywire.ai |

Any processed adjacency matrices created locally under `data/processed/` or
`data/subcircuits/` inherit the source dataset's terms. Derivation
(subsampling, isolated-node removal, adjacency construction) does not change
the underlying license.

## Original publications to cite

- White, J. G., Southgate, E., Thomson, J. N., & Brenner, S. (1986). The
  structure of the nervous system of the nematode *Caenorhabditis elegans*.
  *Philosophical Transactions of the Royal Society B*.
- Cook, S. J., et al. (2019). Whole-animal connectomes of both
  *Caenorhabditis elegans* sexes. *Nature*.
- Ryan, K., Lu, Z., & Meinertzhagen, I. A. (2016). The CNS connectome of a
  tadpole larva of *Ciona intestinalis* (L.) highlights sidedness in the brain
  of a chordate sibling. *eLife*, doi:10.7554/eLife.16962.
- Winding, M., et al. (2023). The connectome of an insect brain. *Science*.
- Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain
  (FlyWire). *Nature*.
- Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome cell
  typing of *Drosophila* (FlyWire).

## Note on FlyWire (non-commercial)

The adult *Drosophila* connectome data originate from FlyWire, which releases
its connectome under a **CC BY-NC 4.0** (attribution, non-commercial) license.
Commercial use of FlyWire-derived data is not permitted under that license,
regardless of the MIT License on this code. Verify the current FlyWire
data-sharing policy before any reuse.
