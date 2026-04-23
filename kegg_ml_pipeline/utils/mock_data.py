from __future__ import annotations

import random


def _gene_group(gene_id: str) -> int:
    return int(gene_id[2]) - 1


def _build_pathway(
    pathway_id: str,
    name: str,
    source: str,
    group_index: int,
    genes_by_group: dict[int, list[str]],
    rng: random.Random,
) -> dict:
    total_genes = rng.randint(10, 25)
    group_genes = int(round(total_genes * 0.7))
    other_genes = total_genes - group_genes

    primary_genes = rng.sample(genes_by_group[group_index], k=group_genes)
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
    rng = random.Random(seed)

    genes = [
        f"AT{chromosome}G{index:05d}"
        for chromosome in range(1, 6)
        for index in range(1, 61)
    ]
    go_terms = [f"GO:{index:07d}" for index in range(1, 151)]
    go_clusters = [go_terms[index : index + 30] for index in range(0, 150, 30)]

    genes_by_group = {group_index: [] for group_index in range(5)}
    for gene in genes:
        genes_by_group[_gene_group(gene)].append(gene)

    gene_go: dict[str, set[str]] = {}
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
        pathways[pathway_id] = _build_pathway(
            pathway_id=pathway_id,
            name=f"Mock AraCyc Pathway {pathway_index}",
            source="AraCyc",
            group_index=pathway_index % 5,
            genes_by_group=genes_by_group,
            rng=rng,
        )

    return {
        "genes": genes,
        "go_terms": go_terms,
        "gene_go": gene_go,
        "pathways": pathways,
    }
