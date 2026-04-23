from __future__ import annotations

import random


def _gene_group(gene_id: str) -> int:
    """Map Arabidopsis gene IDs to one of five coarse functional groups.

    In the mock universe we intentionally align each chromosome prefix
    (`AT1G` .. `AT5G`) with one latent group so the downstream model has
    a learnable but not perfectly clean signal.
    """
    return int(gene_id[2]) - 1


def _build_pathway(
    pathway_id: str,
    name: str,
    source: str,
    group_index: int,
    genes_by_group: dict[int, list[str]],
    rng: random.Random,
) -> dict:
    """Build one synthetic pathway with a 70/30 in-group/out-group mix.

    The pathway contains mostly genes from its assigned latent group, plus
    a smaller fraction of genes sampled from the remaining groups. This
    mirrors real pathway data loosely: coherent core membership with noise.
    """
    total_genes = rng.randint(10, 25)
    group_genes = int(round(total_genes * 0.7))
    other_genes = total_genes - group_genes

    # The majority of pathway members come from the pathway's own group.
    primary_genes = rng.sample(genes_by_group[group_index], k=group_genes)

    # The remaining genes add cross-group contamination so the mock task is
    # not perfectly separable.
    secondary_pool = [
        gene
        for other_group, group_genes_list in genes_by_group.items()
        if other_group != group_index
        for gene in group_genes_list
    ]
    secondary_genes = rng.sample(secondary_pool, k=other_genes) if other_genes else []

    return {
        "name": name,
        "genes": set(primary_genes + secondary_genes),
        "source": source,
    }


def generate_mock_universe(seed: int = 42) -> dict:
    """Generate a reproducible synthetic universe shared across mock runs.

    Returned structure:
    - genes: 300 Arabidopsis-like gene IDs across five chromosome groups
    - go_terms: 150 GO-like terms split into five latent clusters
    - gene_go: per-gene GO annotations with 70% from the matched cluster
    - pathways: 35 synthetic pathways with the same 70/30 group structure

    The exact ratios are chosen to create a meaningful signal for the
    baseline XGBoost model while still leaving overlap between classes.
    """
    # Use a dedicated Random instance so the function is deterministic and
    # does not leak randomness into the global interpreter state.
    rng = random.Random(seed)

    # Build 300 genes: 5 chromosomes x 60 genes per chromosome.
    genes = [
        f"AT{chromosome}G{index:05d}"
        for chromosome in range(1, 6)
        for index in range(1, 61)
    ]

    # Build 150 GO terms and partition them into five clusters of 30 terms.
    # Each cluster acts like a latent biological theme tied to one group.
    go_terms = [f"GO:{index:07d}" for index in range(1, 151)]
    go_clusters = [go_terms[index : index + 30] for index in range(0, 150, 30)]

    # Pre-index genes by group for efficient pathway sampling later.
    genes_by_group = {group_index: [] for group_index in range(5)}
    for gene in genes:
        genes_by_group[_gene_group(gene)].append(gene)

    gene_go: dict[str, set[str]] = {}

    # For each group, also precompute the union of all "other" clusters.
    # This makes the 30% cross-cluster sampling rule straightforward.
    other_cluster_terms = {
        group_index: [
            go_term
            for other_group, cluster in enumerate(go_clusters)
            if other_group != group_index
            for go_term in cluster
        ]
        for group_index in range(5)
    }

    for gene in genes:
        group_index = _gene_group(gene)
        total_terms = rng.randint(5, 15)

        # The mock signal is enforced here:
        # 70% of GO annotations come from the gene's matched cluster and 30%
        # come from the remaining clusters.
        same_cluster_terms = int(round(total_terms * 0.7))
        other_cluster_count = total_terms - same_cluster_terms

        selected_terms = rng.sample(go_clusters[group_index], k=same_cluster_terms)
        if other_cluster_count:
            selected_terms.extend(
                rng.sample(other_cluster_terms[group_index], k=other_cluster_count)
            )

        gene_go[gene] = set(selected_terms)

    pathways: dict[str, dict] = {}
    for pathway_index in range(30):
        pathway_id = f"ath_mock_{pathway_index:02d}"

        # KEGG mock pathways dominate the synthetic universe because KEGG is
        # the main target source in this project.
        pathways[pathway_id] = _build_pathway(
            pathway_id=pathway_id,
            name=f"Mock KEGG Pathway {pathway_index:02d}",
            source="KEGG",
            group_index=pathway_index % 5,
            genes_by_group=genes_by_group,
            rng=rng,
        )

    for pathway_index in range(5):
        pathway_id = f"aracyc:mock_{pathway_index}"

        # A smaller AraCyc slice is included so later merge logic can be
        # tested against multiple pathway sources.
        pathways[pathway_id] = _build_pathway(
            pathway_id=pathway_id,
            name=f"Mock AraCyc Pathway {pathway_index}",
            source="AraCyc",
            group_index=pathway_index % 5,
            genes_by_group=genes_by_group,
            rng=rng,
        )

    # The returned object is intentionally plain and serializable after
    # converting sets via utils.io_utils.save_json.
    return {
        "genes": genes,
        "go_terms": go_terms,
        "gene_go": gene_go,
        "pathways": pathways,
    }
