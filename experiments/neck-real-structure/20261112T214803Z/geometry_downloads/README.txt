This bucket contains public data related to the Brain And Nerve Cord connectome of an adult female Drosophila melanogaster fly. The project's homepage is https://banc.community, where you can find lots of additional information.

If you run into any trouble when trying to use the files you see here, the best way to get help is to open an issue on our GitHub repository: https://github.com/jasper-tms/the-BANC-fly-connectome/issues/new

For per-product detail — exact column schemas, coordinate units, provenance, and known gotchas — see the documentation/ folder, which holds one markdown note per data product. This README is the index; documentation/ is the depth.

These data are also mirrored, with a citable DOI and the same per-product documentation, in the Harvard Dataverse: https://doi.org/10.7910/DVN/7WTH1N. Please cite that DOI when referring to the dataset as a whole; cite the relevant Zenodo DOI (listed at the end of this README) when referring to a specific piece of analysis software.


Files/folders in this bucket serve two distinct purposes:

Category 1: Datasets intended for users to download and interact with via their choice of tools (e.g. programming languages)
- BANC-to-BANC_top_NBLAST_scores.csv: Similarity scores between BANC neurons, allowing users to see all neurons that are morphologically similar to any given BANC neuron of interest. Each row in the csv file lists all the best matches for a single query neuron. Between 100 and 1000 matches are shown for each query neuron, with the list ending after NBLAST scores go below 0 (unless the scores go below 0 before 100 neurons have been listed, in which case we continue the list to the 100th match). This is the convenient "best matches per neuron" digest; the full pairwise tables are under nblast/.
- nblast: Pairwise NBLAST morphological-similarity tables (Arrow .feather), one per comparison. banc_native_nblast.feather is BANC-vs-BANC; banc_mirror_nblast.feather is BANC vs its own mirror image (for left/right pairing); and banc_{fafb_783,manc_v1.2.1,hemibrain_v1.2.1,malecns_v0.9,fanc_1116}_nblast.feather compare BANC against each reference connectome. Each table is long (one row per query-vs-candidate pair). The similarity value is the `score` column (normalised mean NBLAST, ~[-1, 1]; >=0.3 loose, >=0.5 solid, >=0.7 strong) — the `root_*` columns (root_626/root_850/root_888) are BANC query IDs, NOT scores. `match_id` is the matched reference neuron's id and `match_cell_type` is that neuron's label as of when the table was built (a frozen snapshot; the live *_meta.feather can differ after re-curation). `validation = TRUE` flags matches that passed expert review. The banc_*_reviewed_matches.csv files are the curated accept/reject decisions. A high score is necessary but not sufficient for a correct cell-type match — use the curated cell_type in banc_888_meta.feather as the identity of record.
- neuron_annotations: Tables containing different types of information about each cell in the connectome. See also https://banc.community/Annotations-(cell-types,-etc.) for some additional description of some of these tables.
  - "codex_annotations" is the most comprehensive annotation table and is likely to be the first table users will want to look through. Among other things, cell type names are listed in this table.
  - "cell_info" has largely been superceded by "codex_annotations". Both contain general information about cells.
  - "backbone_proofread" indicates which cells have been marked as proofread.
  - "cell_representative_point" contains a single point for each cell that has been paid attention to so far (e.g. annotated and/or proofread), which is most but not yet all of the cells in the dataset.
  - "neck_connective_y92500" contains a point in each axon traversing the neck connective. All ascending and descending neurons have an entry in this table.
  - "peripheral nerves" contains a point in each axon traversing each peripheral nerve. The name of the nerve is indicated with each point.
  - "somas_v1" contains a point in each nucleus.
- neuron_colormips: Depth-colored maximum intensity projection images of proofread neurons from the EM dataset. These can be used in tools like NeuronBridge to search for genetic driver lines that label neurons you're interested in.
- neuron_connectivity: Information about which neurons make how many synapses in what locations with which other neurons. Note that there are currently two synapse tables, synapses_v1 and synapses_250226. We are in the process of finishing some data handling steps that will allow us to fully migrate to using synapses_250226, after which synapses_v1 will be removed. (Currently, we have not yet assigned each synapse in synapses_250226 to the region of the nervous system it is located in, so you may wish to temporarily use the synapses_v1 files that include this region information if that information is important to you.)
- neuron_skeletons: Skeleton representations of proofread neurons from the EM dataset. The bucket-root neuron_skeletons.zip is the low-resolution set generated by the python package `pcg_skel` (L2 chunked-graph skeletons). Higher-resolution skeletons, and skeletons for the other datasets warped into BANC space, live under compiled_data/ — see that entry below.
- compiled_data: Per-neuron tables that mirror the BANC framework across the five datasets we work with: banc_888/, fafb_783/, manc_121/, hemibrain_121/, malecns_09/. All files use unified column names (most importantly root_id, super_class, cell_class, cell_sub_class, cell_type, hemilineage, region, side, flow), so the same analysis code reads any dataset. Bates, Phelps, Kim, Yang et al., 2026 (the BANC paper) use synapses v2 (size >= 5) throughout; the v3 variants (size >= 10) are the newer detector recommended for new work.

  Key files for analysing the BANC connectome:
  - banc_888/banc_888_meta.feather: per-neuron metadata table; primary join target for all downstream analysis (188k+ rows, ~80 columns).
  - banc_888/banc_888_metrics.feather: slim morphology + synapse-count table (l2_cable_length_um, volume_nm3, input_connections, output_connections, input_side_index, output_side_index, mitochondria, mitochondria_volume, pd_width, segregation_index); ~10x faster to load than meta when only morphology is needed.
  - banc_888/banc_888_edgelist_simple_v2.feather: neuron-to-neuron synaptic edgelist (synapses_v2, size >= 5). Columns: pre, post, count, norm, pre_count, post_count. A v3 variant (size >= 10) is also available.
  - banc_888/banc_888_synapses_v2_enriched.parquet: per-synapse table joined with cell-type annotations; useful for region-resolved or partner-resolved analyses.
  - banc_888/banc_888_neurotransmitter_prediction_v2.csv: per-neuron neurotransmitter predictions with confidence scores.
  - banc_888/banc_888_betweenness_{all_to_all,afferent_to_efferent}_v2.csv: graph-centrality measures.
  - banc_888/banc_888_cns_network_spectral_clustering_v2.csv: neuron memberships of the spectral CNS networks.
  - banc_888/influence/all_to_all/: pre-computed multiplicative influence scores between all proofread neuron pairs (parquet directory; ~287 GB).

  Per-neuron SWC skeletons (subfolders ending in _swc):
  - banc_888/banc_banc_space_swc/: BANC neurons, two resolutions in one folder, distinguished by filename suffix. <root_id>_skeleton.swc is the full-resolution skeleton built from the segmentation mesh with the python package `skeletor`; <root_id>_l2.swc is the coarse L2 skeleton from `pcg_skel`. The export rule is "use _skeleton if it exists, else _l2". The paper's NBLAST and most natverse work used the L2 set, so use _l2 for like-for-like reproduction and _skeleton for finest morphology.
  - {fafb_783,malecns_09,manc_121}/*_banc_space_swc/: skeletons of each reference connectome warped into BANC space (the morphology behind the matching nblast/ tables). These are keyed only by the source dataset's neuron id with NO _l2/_skeleton suffix and mix resolutions; tell them apart by node spacing (e.g. NAVis sampling_resolution), not by filename. Each dataset's native-space skeletons sit alongside in *_<dataset>_space_swc/.
  - All SWC node coordinates are in NANOMETRES in BANC space (not voxels).

  Column conventions (apply to all datasets):
  - root_id: dataset's canonical neuron identifier (also written as banc_888_id, fafb_783_id, etc.).
  - super_class: top-level category - sensory, ascending, descending, intrinsic, motor, visceral_circulatory, ascending_visceral_circulatory, optic.
  - cell_class / cell_sub_class / cell_type: increasingly specific labels.
  - region: central_brain, optic_lobe, ventral_nerve_cord, or cervical_connective.
  - side: left, right, or center.
  - flow: afferent, intrinsic, or efferent.
  - hemilineage: developmental hemilineage where assigned.
- behavior: Pre-EM tethered-walking behavioural recordings of the BANC fly.
- documentation: Per-product schema notes (one markdown file per data product) covering exact columns, coordinate units, provenance, and gotchas. Start here when a column or file is unclear.

Category 2: Datasets intended for visualization via the web app Neuroglancer, all of which can be seen most easily by opening https://ng.banc.community/view in your browser.
- imported_meshes: Neuron meshes from other connectome datasets warped into alignment with the Brain And Nerve Cord for inter-fly comparisons of specific neuron types. Subfolders are named <dataset>_meshes_<registration>_<date>, e.g. fafb_783_meshes_elastix_tpsreg_240721 = FAFB v783 meshes registered into BANC space with elastix + thin-plate-spline, build date 2024-07-21.
- microCT: An X-ray micro-computed tomography scan of the fixed, stained, and resin-embedded nervous system before serial sectioning for electron microscopy.
- light_level: Light-microscopy image data associated with the project.
- neuron_meshes: Neuron meshes from the BANC.
- nuclei: Nucleus segmentation of the EM dataset.
- region_outlines: Meshes outlining the various regions of the nervous system, both broad regions (brain vs VNC) and fine regions as delineated by previous papers (mostly from Ito et al. 2014 Neuron for the brain and Court et al. 2020 Neuron for the VNC).
- synapses: Synapse segmentation of the EM dataset.
- registrations: Elastix/thin-plate-spline registration fields bridging BANC to the standard JRC2018F (brain) and JRC2018VNC (nerve cord) template atlases; the transforms behind the warped skeletons and imported_meshes above.
- templates: Synapse density maps used for registering the Brain And Nerve Cord dataset to the standard brain and VNC atlases.

It is less likely that users will want to directly download any of these files in category 2. If you do want to interact with these category 2 datasets programmatically, the best way to do so is via the Python package CloudVolume: https://github.com/seung-lab/cloud-volume

(_archive/ holds superseded files kept for provenance; it is not intended for general use.)


Code and citation (Zenodo)

The analysis software and curated ground-truth datasets behind this dataset are archived on Zenodo; please cite the relevant Zenodo DOI when you use a component (byte-for-byte snapshots are also mirrored in the Harvard Dataverse deposit under code/):
- bancr — R client for accessing BANC data: https://doi.org/10.5281/zenodo.20350647
- bancpipeline — R + SLURM pipeline that builds the compiled_data/ products: https://doi.org/10.5281/zenodo.20350571
- BANC-project — R analysis + figure code for the paper: https://doi.org/10.5281/zenodo.20350641
- influencer — R influence-score wrapper: https://doi.org/10.5281/zenodo.20350563
- ConnectomeInfluenceCalculator — Python influence engine: https://doi.org/10.5281/zenodo.15999929
- nat.ggplot — 2D neuron / mesh rendering for ggplot2: https://doi.org/10.5281/zenodo.20350565
- synister_banc — neurotransmitter-prediction code + ground truth: https://doi.org/10.5281/zenodo.20350569
- drosophila_neurotransmitters — curated literature ground-truth of transmitter usage by cell type: https://doi.org/10.5281/zenodo.20818141
- drosophila_neuropeptides — curated literature ground-truth of neuropeptide expression by cell type: https://doi.org/10.5281/zenodo.20818143

the-BANC-fly-connectome (Jasper Phelps's Python toolkit, https://github.com/jasper-tms/the-BANC-fly-connectome) and fly_connectome_data_tutorial do not yet have minted Zenodo DOIs; cite the GitHub repositories until they do.
