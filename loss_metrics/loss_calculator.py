from dgh.dgh import DGH, most_recent_common_ancestor


def calculate_md_loss(
    DGHs: list[DGH],
    raw_dataset: list[dict[str, str]],
    anonymized_dataset: list[dict[str, str]],
) -> int:
    """
    Calculate the Mondrian distance loss between the raw and anonymized datasets
    using the provided Domain Generalization Hierarchies (DGHs).

    Args:
        DGHs (list[DGH]): List of DGHs to use for calculating the loss.
        raw_dataset (list[dict[str, str]]): The original dataset.
        anonymized_dataset (list[dict[str, str]]): The anonymized dataset.

    Returns:
        int: The calculated Mondrian distance loss.
    """
    total_loss = 0
    for dgh in DGHs:
        # Calculate the loss for each DGH
        dgh_loss = calculate_md_loss_for_column(dgh, raw_dataset, anonymized_dataset)
        total_loss += dgh_loss

    return total_loss


def calculate_md_loss_for_column(
    dgh: DGH,
    raw_dataset: list[dict[str, str]],
    anonymized_dataset: list[dict[str, str]],
) -> int:
    total_loss = 0
    col = dgh.column_name
    # build mapping from raw row index to anonymized row index
    # (assume same ordering or use a unique key to align records)
    for raw_row, anon_row in zip(raw_dataset, anonymized_dataset):
        raw_val = raw_row[col]
        anon_val = anon_row[col]
        raw_node = dgh.find_node_by_value(raw_val)
        anon_node = dgh.find_node_by_value(anon_val)
        if raw_node is None or anon_node is None:
            continue
        raw_depth = raw_node.depth()
        anon_depth = anon_node.depth()
        dist = raw_depth - anon_depth
        total_loss += dist
    return total_loss


def calculate_lm_loss(
    DGHs: list[DGH],
    anonymized_dataset: list[dict[str, str]],
) -> float:
    """
    Calculate the LM distance loss between the raw and anonymized datasets
    using the provided Domain Generalization Hierarchies (DGHs).

    Args:
        DGHs (list[DGH]): List of DGHs to use for calculating the loss.
        anonymized_dataset (list[dict[str, str]]): The anonymized dataset.

    Returns:
        float: The calculated LM distance loss.
    """

    total_loss = 0.0

    for dgh in DGHs:
        # Calculate the loss for each DGH
        dgh_loss = calculate_lm_loss_for_column(dgh, anonymized_dataset)
        total_loss += dgh_loss

    return total_loss


def calculate_lm_loss_for_column(
    dgh: DGH,
    anonymized_dataset: list[dict[str, str]],
) -> float:
    """
    LM(val) = (leaf_count(val) – 1) / (total_leaves – 1)
    Dataset loss = weighted sum of LM(val) for each anonymized column
    """
    total_leaves = dgh.root.leaf_count()
    assert total_leaves >= 1, "DGH must have at least one leaf node."

    denominator = total_leaves - 1
    col = dgh.column_name

    loss = 0.0
    for anon_row in anonymized_dataset:
        anon_val = anon_row[col]
        node = dgh.find_node_by_value(anon_val)
        if node is None:
            continue
        lm = (node.leaf_count() - 1) / denominator
        loss += lm
    return loss
