#!/usr/bin/env python3


from model.data_loading import *  

from model.data_loading import (
    _tensor_state_dict,
    _extract_state_dict,
    _select_encoder_prefix,
    _normalize_self_encoder_state,
    _checkpoint_metadata,
    _find_metadata_value,
    _infer_transformer_architecture,
    _load_self_transformer_spec,
    _joined_row_text,
    _stable_beta_node_id,
    _make_beta_key,
    _coverage_report,
)

from model.data_splitting import (
    DisjointSet,
    load_supervision_pool,
    attach_donor_metadata,
    build_beta_component_groups,
    prepare_validation_pool,
    _donor_label_strata,
    _subset_distribution_error,
    _partition_balance_score,
    _make_donor_balanced_group_folds,
    make_group_outer_folds,
    make_group_train_val_split,
    summarize_split_frame,
    validate_group_disjointness,
    validate_strict_cdr3b_disjointness,
    save_split_audit,
    load_existing_group5_fold,
    add_protocol_columns,
    aggregate_fold_metrics,
)

from model.training import (
    InspectableTransformerEncoderLayer,
    SelfSequenceEncoder,
    masked_mean_pool,
    SelfTransformerSequenceEncoder,
    TCRNodeEncoder,
    PeptideNodeEncoder,
    mean_aggregate_global,
    grouped_softmax,
    uniform_group_weights,
    RelationAwareEdgeAttention,
    GateConfiguration,
    parse_gate_configuration,
    probability_to_logit,
    optimizer_group_lr,
    EdgeDecoder,
    AuxiliaryEdgeDecoder,
    SelfPretrainedGraphModel,
    sample_auxiliary_loss,
    make_stage2_optimizer,
    get_pos_weight,
    evaluate_stage2,
    save_predictions,
    _attention_concentration,
    _empty_interpretability_frame,
    export_graph_attention,
    _transformer_attention_statistics,
    export_transformer_attention,
    resolve_existing_stage2_checkpoint,
    load_stage2_checkpoint_compat,
    _external_checkpoint_base_dir,
    resolve_external_checkpoint,
    _checkpoint_saved_args,
    _external_checkpoint_mode_settings,
    _copy_saved_architecture_args,
    load_external_stage2_checkpoint_compat,
    run_external_inference,
    train_one_ablation,
)

from model.ablation import (
    run_model_grid,
    run_existing_group5_protocol,
    run_group5_protocol,
    run_lodo_protocol,
    run_original_protocol,
    parse_args,
    validate_args,
    main,
)

if __name__ == "__main__":
    main()
