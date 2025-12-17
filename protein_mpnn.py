"""
ProteinMPNN wrapper class.

This module provides a clean API for using ProteinMPNN without the command-line interface.
After pip install, you can use it as:

    from protein_mpnn import ProteinMPNN

    model = ProteinMPNN(model_name='v_48_020', device='cuda')
    results = model.sample(pdb_path='protein.pdb', num_seq_per_target=10, sampling_temp='0.1')
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
import os
import warnings

from protein_mpnn_utils import (
    loss_nll,
    loss_smoothed,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
    cat_neighbors_nodes,
    _scores,
    _S_to_seq,
    tied_featurize,
    parse_PDB,
    parse_fasta,
    StructureDataset,
    StructureDatasetPDB,
    ProteinMPNN as ProteinMPNNModel,
)


class ProteinMPNN:
    """
    Wrapper class for ProteinMPNN model that separates model loading from inference.

    This class loads the model once during initialization and provides methods for
    sequence generation, scoring, and probability calculation.

    Args:
        model_name: Name of the model weights (default: 'v_48_020')
        path_to_model_weights: Path to directory containing model weights (default: auto-detect)
        ca_only: Use CA-only model (default: False)
        use_soluble_model: Use soluble protein model (default: False)
        device: Device to run on ('cuda' or 'cpu', default: auto-detect)
        suppress_print: Suppress printing (default: False)
        compile_model: Compile model with torch.compile (default: None = auto-detect, True = force, False = disable)
    """

    def __init__(
        self,
        model_name="v_48_020",
        path_to_model_weights=None,
        ca_only=False,
        use_soluble_model=False,
        device=None,
        suppress_print=False,
        compile_model=None,
    ):
        self.model_name = model_name
        self.ca_only = ca_only
        self.use_soluble_model = use_soluble_model
        self.suppress_print = suppress_print

        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model architecture parameters
        self.hidden_dim = 128
        self.num_layers = 3

        # Determine model folder path
        if path_to_model_weights:
            model_folder_path = path_to_model_weights
            if model_folder_path[-1] != "/":
                model_folder_path = model_folder_path + "/"
        else:
            file_path = os.path.realpath(__file__)
            k = file_path.rfind("/")
            if ca_only:
                if not self.suppress_print:
                    print("Using CA-ProteinMPNN!")
                model_folder_path = file_path[:k] + "/ca_model_weights/"
                if use_soluble_model:
                    raise ValueError("CA-SolubleMPNN is not available yet")
            else:
                if use_soluble_model:
                    if not self.suppress_print:
                        print("Using ProteinMPNN trained on soluble proteins only!")
                    model_folder_path = file_path[:k] + "/soluble_model_weights/"
                else:
                    model_folder_path = file_path[:k] + "/vanilla_model_weights/"

        # Load checkpoint
        checkpoint_path = model_folder_path + f"{model_name}.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.noise_level = checkpoint["noise_level"]
        self.num_edges = checkpoint["num_edges"]

        # Initialize model
        self.model = ProteinMPNNModel(
            ca_only=ca_only,
            num_letters=21,
            node_features=self.hidden_dim,
            edge_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            augment_eps=0.0,  # Will be set per-call if needed
            k_neighbors=self.num_edges,
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Compile model if requested
        if compile_model is None:
            # Auto-detect: compile if PyTorch 2.0+ and CUDA
            compile_model = (
                hasattr(torch, 'compile') and
                self.device.type == 'cuda' and
                torch.cuda.is_available()
            )

        if compile_model:
            if not hasattr(torch, 'compile'):
                if not self.suppress_print:
                    warnings.warn("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
            else:
                if not self.suppress_print:
                    print("Compiling model with torch.compile (first call will be slower)...")
                self.model = torch.compile(self.model, mode='reduce-overhead')

        if not self.suppress_print:
            print(40 * "-")
            print("Number of edges:", self.num_edges)
            print(f"Training noise level: {self.noise_level}A")

        # Amino acid alphabet
        self.alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        self.alphabet_dict = dict(zip(self.alphabet, range(21)))

    def _load_structure(
        self,
        pdb_path_or_str=None,
        jsonl_path=None,
        pdb_path_chains=None,
        chain_id_dict=None,
        max_length=200000,
    ):
        """Load structure from PDB file, PDB string content, or JSONL file.

        Args:
            pdb_path_or_str: Path to PDB file OR PDB file content as string
            jsonl_path: Path to JSONL file
            pdb_path_chains: Space-separated chain IDs to design
            chain_id_dict: Dictionary specifying designed/fixed chains
            max_length: Maximum sequence length
        """
        if pdb_path_or_str:
            pdb_dict_list = parse_PDB(pdb_path_or_str, ca_only=self.ca_only)
            dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)
            all_chain_list = [
                item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == "seq_chain"
            ]

            if pdb_path_chains:
                designed_chain_list = [str(item) for item in pdb_path_chains.split()]
            else:
                designed_chain_list = all_chain_list

            fixed_chain_list = [
                letter for letter in all_chain_list if letter not in designed_chain_list
            ]
            chain_id_dict = {pdb_dict_list[0]["name"]: (designed_chain_list, fixed_chain_list)}
        elif jsonl_path:
            dataset = StructureDataset(
                jsonl_path,
                truncate=None,
                max_length=max_length,
                verbose=not self.suppress_print,
            )
        else:
            raise ValueError("Either pdb_path_or_str or jsonl_path must be provided")

        return dataset, chain_id_dict

    def to_device(self, device):
        """Move model to specified device.

        Args:
            device: Device to move to ('cuda', 'cpu', or torch.device object)

        Returns:
            self (for method chaining)
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model.to(device)
        return self

    def _get_optimal_batch_size(self, num_seq_per_target):
        """Calculate optimal batch size based on available GPU memory.

        Args:
            num_seq_per_target: Number of sequences to generate

        Returns:
            Recommended batch size
        """
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            return 1

        # Get available GPU memory in GB
        try:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            allocated_memory = torch.cuda.memory_allocated(self.device) / 1e9
            available_memory = total_memory - allocated_memory
        except:
            return 1

        # Conservative batch size selection based on available memory
        # These are empirically determined for typical protein sequences
        if available_memory > 20:  # > 20GB available
            recommended_batch_size = 32
        elif available_memory > 10:  # > 10GB available
            recommended_batch_size = 16
        elif available_memory > 5:  # > 5GB available
            recommended_batch_size = 8
        elif available_memory > 2:  # > 2GB available
            recommended_batch_size = 4
        else:
            recommended_batch_size = 1

        # Don't exceed num_seq_per_target
        recommended_batch_size = min(recommended_batch_size, num_seq_per_target)

        return recommended_batch_size

    def sample(
        self,
        pdb_path_or_str=None,
        jsonl_path=None,
        pdb_path_chains=None,
        num_seq_per_target=1,
        batch_size=1,
        sampling_temp="0.1",
        seed=None,
        chain_id_dict=None,
        fixed_positions_dict=None,
        omit_AAs="X",
        omit_AA_dict=None,
        bias_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        pssm_multi=0.0,
        pssm_threshold=0.0,
        pssm_log_odds_flag=False,
        pssm_bias_flag=False,
        bias_by_res_dict=None,
        backbone_noise=0.0,
        max_length=200000,
    ):
        """
        Generate protein sequences for given structure(s).

        Args:
            pdb_path_or_str: Path to PDB file OR PDB file content as string
            jsonl_path: Path to JSONL file with parsed structures
            pdb_path_chains: Space-separated chain IDs to design (for PDB input)
            num_seq_per_target: Number of sequences to generate per target
            batch_size: Batch size for generation
            sampling_temp: Sampling temperature(s) as string (e.g., "0.1" or "0.1 0.2 0.5")
            seed: Random seed (if None, will be random)
            chain_id_dict: Dictionary specifying designed/fixed chains
            fixed_positions_dict: Dictionary with fixed positions
            omit_AAs: Amino acids to omit (default: 'X')
            omit_AA_dict: Per-position amino acids to omit
            bias_AA_dict: Amino acid composition bias
            tied_positions_dict: Dictionary with tied positions
            pssm_dict: PSSM constraints
            pssm_multi: PSSM weight (0.0-1.0)
            pssm_threshold: PSSM threshold
            pssm_log_odds_flag: Use PSSM log odds
            pssm_bias_flag: Use PSSM bias
            bias_by_res_dict: Per-residue bias
            backbone_noise: Backbone noise level
            max_length: Maximum sequence length

        Returns:
            Dictionary with results for each protein:
            {
                'protein_name': {
                    'native_sequence': str,
                    'sequences': list of str,
                    'scores': np.array,
                    'global_scores': np.array,
                    'seq_recovery_rates': np.array,
                    'sampling_temperatures': list of float,
                    'seed': int
                }
            }
        """
        # Set random seed
        if seed is None:
            seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Smart batch size selection
        if batch_size == 1 and num_seq_per_target > 1:
            optimal_batch_size = self._get_optimal_batch_size(num_seq_per_target)
            if optimal_batch_size > 1:
                batch_size = optimal_batch_size
                if not self.suppress_print:
                    available_mem = torch.cuda.get_device_properties(self.device).total_memory / 1e9 if self.device.type == 'cuda' else 0
                    print(f"[ProteinMPNN] Auto-selected batch_size={batch_size} based on {available_mem:.1f}GB GPU memory (use batch_size parameter to override)")

        # Parse temperatures
        temperatures = [float(item) for item in sampling_temp.split()]

        # Parse omit AAs
        omit_AAs_list = omit_AAs
        omit_AAs_np = np.array([AA in omit_AAs_list for AA in self.alphabet]).astype(
            np.float32
        )

        # Parse bias AAs
        bias_AAs_np = np.zeros(len(self.alphabet))
        if bias_AA_dict:
            for n, AA in enumerate(self.alphabet):
                if AA in list(bias_AA_dict.keys()):
                    bias_AAs_np[n] = bias_AA_dict[AA]

        # Load structure
        dataset, chain_id_dict = self._load_structure(
            pdb_path_or_str=pdb_path_or_str,
            jsonl_path=jsonl_path,
            pdb_path_chains=pdb_path_chains,
            chain_id_dict=chain_id_dict,
            max_length=max_length,
        )

        NUM_BATCHES = num_seq_per_target // batch_size
        BATCH_COPIES = batch_size

        # Update model's backbone noise
        self.model.augment_eps = backbone_noise

        results = {}

        with torch.no_grad():
            for ix, protein in enumerate(dataset):
                batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]

                # Featurize
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding_all,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                ) = tied_featurize(
                    batch_clones,
                    self.device,
                    chain_id_dict,
                    fixed_positions_dict,
                    omit_AA_dict,
                    tied_positions_dict,
                    pssm_dict,
                    bias_by_res_dict,
                    ca_only=self.ca_only,
                )

                pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float()
                name_ = batch_clones[0]["name"]

                # Get native score
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs = self.model(
                    X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1
                )
                mask_for_loss = mask * chain_M * chain_M_pos
                scores = _scores(S, log_probs, mask_for_loss)
                native_score = scores.cpu().data.numpy()
                global_scores = _scores(S, log_probs, mask)
                global_native_score = global_scores.cpu().data.numpy()

                # Storage for generated sequences
                all_sequences = []
                all_scores = []
                all_global_scores = []
                all_seq_recovery = []
                all_temperatures = []

                # Generate sequences
                for temp in temperatures:
                    for j in range(NUM_BATCHES):
                        randn_2 = torch.randn(chain_M.shape, device=X.device)

                        if tied_positions_dict is None:
                            sample_dict = self.model.sample(
                                X,
                                randn_2,
                                S,
                                chain_M,
                                chain_encoding_all,
                                residue_idx,
                                mask=mask,
                                temperature=temp,
                                omit_AAs_np=omit_AAs_np,
                                bias_AAs_np=bias_AAs_np,
                                chain_M_pos=chain_M_pos,
                                omit_AA_mask=omit_AA_mask,
                                pssm_coef=pssm_coef,
                                pssm_bias=pssm_bias,
                                pssm_multi=pssm_multi,
                                pssm_log_odds_flag=bool(pssm_log_odds_flag),
                                pssm_log_odds_mask=pssm_log_odds_mask,
                                pssm_bias_flag=bool(pssm_bias_flag),
                                bias_by_res=bias_by_res_all,
                            )
                            S_sample = sample_dict["S"]
                        else:
                            sample_dict = self.model.tied_sample(
                                X,
                                randn_2,
                                S,
                                chain_M,
                                chain_encoding_all,
                                residue_idx,
                                mask=mask,
                                temperature=temp,
                                omit_AAs_np=omit_AAs_np,
                                bias_AAs_np=bias_AAs_np,
                                chain_M_pos=chain_M_pos,
                                omit_AA_mask=omit_AA_mask,
                                pssm_coef=pssm_coef,
                                pssm_bias=pssm_bias,
                                pssm_multi=pssm_multi,
                                pssm_log_odds_flag=bool(pssm_log_odds_flag),
                                pssm_log_odds_mask=pssm_log_odds_mask,
                                pssm_bias_flag=bool(pssm_bias_flag),
                                tied_pos=tied_pos_list_of_lists_list[0],
                                tied_beta=tied_beta,
                                bias_by_res=bias_by_res_all,
                            )
                            S_sample = sample_dict["S"]

                        # Compute scores
                        log_probs = self.model(
                            X,
                            S_sample,
                            mask,
                            chain_M * chain_M_pos,
                            residue_idx,
                            chain_encoding_all,
                            randn_2,
                            use_input_decoding_order=True,
                            decoding_order=sample_dict["decoding_order"],
                        )
                        mask_for_loss = mask * chain_M * chain_M_pos
                        scores = _scores(S_sample, log_probs, mask_for_loss)
                        scores_np = scores.cpu().data.numpy()

                        global_scores = _scores(S_sample, log_probs, mask)
                        global_scores_np = global_scores.cpu().data.numpy()

                        for b_ix in range(BATCH_COPIES):
                            masked_chain_length_list = masked_chain_length_list_list[b_ix]
                            masked_list = masked_list_list[b_ix]

                            seq_recovery_rate = torch.sum(
                                torch.sum(
                                    torch.nn.functional.one_hot(S[b_ix], 21)
                                    * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                                    axis=-1,
                                )
                                * mask_for_loss[b_ix]
                            ) / torch.sum(mask_for_loss[b_ix])

                            seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                            score = scores_np[b_ix]
                            global_score = global_scores_np[b_ix]

                            # Reorder sequence by chain
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(seq[start:end])
                                start = end

                            seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                            l0 = 0
                            for mc_length in list(
                                np.array(masked_chain_length_list)[np.argsort(masked_list)]
                            )[:-1]:
                                l0 += mc_length
                                seq = seq[:l0] + "/" + seq[l0:]
                                l0 += 1

                            all_sequences.append(seq)
                            all_scores.append(score)
                            all_global_scores.append(global_score)
                            all_seq_recovery.append(seq_recovery_rate.cpu().numpy())
                            all_temperatures.append(temp)

                # Get native sequence
                native_seq = _S_to_seq(S[0,], chain_M[0,])
                masked_chain_length_list = masked_chain_length_list_list[0]
                masked_list = masked_list_list[0]

                start = 0
                end = 0
                list_of_AAs = []
                for mask_l in masked_chain_length_list:
                    end += mask_l
                    list_of_AAs.append(native_seq[start:end])
                    start = end

                native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                l0 = 0
                for mc_length in list(
                    np.array(masked_chain_length_list)[np.argsort(masked_list)]
                )[:-1]:
                    l0 += mc_length
                    native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                    l0 += 1

                results[name_] = {
                    "native_sequence": native_seq,
                    "sequences": all_sequences,
                    "scores": np.array(all_scores),
                    "global_scores": np.array(all_global_scores),
                    "seq_recovery_rates": np.array(all_seq_recovery),
                    "sampling_temperatures": all_temperatures,
                    "native_score": native_score.mean(),
                    "global_native_score": global_native_score.mean(),
                    "seed": seed,
                    "designed_chains": masked_list_list[0],
                    "fixed_chains": visible_list_list[0],
                }

        return results

    def score(
        self,
        pdb_path_or_str=None,
        jsonl_path=None,
        pdb_path_chains=None,
        fasta_path=None,
        num_batches=1,
        seed=None,
        chain_id_dict=None,
        fixed_positions_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        backbone_noise=0.0,
        max_length=200000,
    ):
        """
        Score backbone-sequence pairs.

        Args:
            pdb_path_or_str: Path to PDB file OR PDB file content as string
            jsonl_path: Path to JSONL file with parsed structures
            pdb_path_chains: Space-separated chain IDs to design (for PDB input)
            fasta_path: Path to FASTA file with sequences to score
            num_batches: Number of batches for scoring
            seed: Random seed
            chain_id_dict: Dictionary specifying designed/fixed chains
            fixed_positions_dict: Dictionary with fixed positions
            omit_AA_dict: Per-position amino acids to omit
            tied_positions_dict: Dictionary with tied positions
            pssm_dict: PSSM constraints
            bias_by_res_dict: Per-residue bias
            backbone_noise: Backbone noise level
            max_length: Maximum sequence length

        Returns:
            Dictionary with results for each protein and sequence:
            {
                'protein_name': {
                    'pdb_scores': {
                        'mean_score': float,
                        'std_score': float,
                        'mean_global_score': float,
                        'std_global_score': float,
                        'scores': np.array,
                        'global_scores': np.array,
                        'sequence': str
                    },
                    'fasta_scores': [
                        {
                            'mean_score': float,
                            'std_score': float,
                            'mean_global_score': float,
                            'std_global_score': float,
                            'scores': np.array,
                            'global_scores': np.array,
                            'sequence': str
                        },
                        ...
                    ]
                }
            }
        """
        # Set random seed
        if seed is None:
            seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load structure
        dataset, chain_id_dict = self._load_structure(
            pdb_path_or_str=pdb_path_or_str,
            jsonl_path=jsonl_path,
            pdb_path_chains=pdb_path_chains,
            chain_id_dict=chain_id_dict,
            max_length=max_length,
        )

        # Update model's backbone noise
        self.model.augment_eps = backbone_noise

        results = {}

        with torch.no_grad():
            for ix, protein in enumerate(dataset):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]

                # Featurize
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding_all,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                ) = tied_featurize(
                    batch_clones,
                    self.device,
                    chain_id_dict,
                    fixed_positions_dict,
                    omit_AA_dict,
                    tied_positions_dict,
                    pssm_dict,
                    bias_by_res_dict,
                    ca_only=self.ca_only,
                )

                name_ = batch_clones[0]["name"]

                loop_c = 0
                fasta_seqs = []
                if fasta_path:
                    fasta_names, fasta_seqs = parse_fasta(fasta_path, omit=["/"])
                    loop_c = len(fasta_seqs)

                protein_results = {"pdb_scores": None, "fasta_scores": []}

                for fc in range(1 + loop_c):
                    native_score_list = []
                    global_native_score_list = []

                    if fc > 0:
                        input_seq_length = len(fasta_seqs[fc - 1])
                        S_input = torch.tensor(
                            [self.alphabet_dict[AA] for AA in fasta_seqs[fc - 1]],
                            device=self.device,
                        )[None, :].repeat(X.shape[0], 1)
                        S[:, :input_seq_length] = S_input

                    for j in range(num_batches):
                        randn_1 = torch.randn(chain_M.shape, device=X.device)
                        log_probs = self.model(
                            X,
                            S,
                            mask,
                            chain_M * chain_M_pos,
                            residue_idx,
                            chain_encoding_all,
                            randn_1,
                        )
                        mask_for_loss = mask * chain_M * chain_M_pos
                        scores = _scores(S, log_probs, mask_for_loss)
                        native_score = scores.cpu().data.numpy()
                        native_score_list.append(native_score)

                        global_scores = _scores(S, log_probs, mask)
                        global_native_score = global_scores.cpu().data.numpy()
                        global_native_score_list.append(global_native_score)

                    native_score = np.concatenate(native_score_list, 0)
                    global_native_score = np.concatenate(global_native_score_list, 0)

                    seq_str = _S_to_seq(S[0,], chain_M[0,])

                    score_dict = {
                        "mean_score": float(native_score.mean()),
                        "std_score": float(native_score.std()),
                        "mean_global_score": float(global_native_score.mean()),
                        "std_global_score": float(global_native_score.std()),
                        "scores": native_score,
                        "global_scores": global_native_score,
                        "sequence": seq_str,
                        "sample_size": native_score.shape[0],
                    }

                    if fc == 0:
                        protein_results["pdb_scores"] = score_dict
                    else:
                        protein_results["fasta_scores"].append(score_dict)

                results[name_] = protein_results

        return results

    def conditional_probs(
        self,
        pdb_path_or_str=None,
        jsonl_path=None,
        pdb_path_chains=None,
        num_batches=1,
        backbone_only=False,
        seed=None,
        chain_id_dict=None,
        fixed_positions_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        backbone_noise=0.0,
        max_length=200000,
    ):
        """
        Calculate conditional probabilities p(s_i | rest of sequence and backbone).

        Args:
            pdb_path_or_str: Path to PDB file OR PDB file content as string
            jsonl_path: Path to JSONL file with parsed structures
            pdb_path_chains: Space-separated chain IDs to design (for PDB input)
            num_batches: Number of batches
            backbone_only: If True, calculate p(s_i | backbone) instead
            seed: Random seed
            chain_id_dict: Dictionary specifying designed/fixed chains
            fixed_positions_dict: Dictionary with fixed positions
            omit_AA_dict: Per-position amino acids to omit
            tied_positions_dict: Dictionary with tied positions
            pssm_dict: PSSM constraints
            bias_by_res_dict: Per-residue bias
            backbone_noise: Backbone noise level
            max_length: Maximum sequence length

        Returns:
            Dictionary with results for each protein:
            {
                'protein_name': {
                    'log_probs': np.array,  # [num_batches, L, 21]
                    'sequence': np.array,   # [L]
                    'mask': np.array,       # [L]
                    'design_mask': np.array # [L]
                }
            }
        """
        # Set random seed
        if seed is None:
            seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load structure
        dataset, chain_id_dict = self._load_structure(
            pdb_path_or_str=pdb_path_or_str,
            jsonl_path=jsonl_path,
            pdb_path_chains=pdb_path_chains,
            chain_id_dict=chain_id_dict,
            max_length=max_length,
        )

        # Update model's backbone noise
        self.model.augment_eps = backbone_noise

        results = {}

        with torch.no_grad():
            for ix, protein in enumerate(dataset):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]

                # Featurize
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding_all,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                ) = tied_featurize(
                    batch_clones,
                    self.device,
                    chain_id_dict,
                    fixed_positions_dict,
                    omit_AA_dict,
                    tied_positions_dict,
                    pssm_dict,
                    bias_by_res_dict,
                    ca_only=self.ca_only,
                )

                name_ = batch_clones[0]["name"]

                log_conditional_probs_list = []
                for j in range(num_batches):
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    log_conditional_probs = self.model.conditional_probs(
                        X,
                        S,
                        mask,
                        chain_M * chain_M_pos,
                        residue_idx,
                        chain_encoding_all,
                        randn_1,
                        backbone_only,
                    )
                    log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())

                concat_log_p = np.concatenate(log_conditional_probs_list, 0)
                mask_out = (chain_M * chain_M_pos * mask)[0,].cpu().numpy()

                results[name_] = {
                    "log_probs": concat_log_p,
                    "sequence": S[0,].cpu().numpy(),
                    "mask": mask[0,].cpu().numpy(),
                    "design_mask": mask_out,
                }

        return results

    def unconditional_probs(
        self,
        pdb_path_or_str=None,
        jsonl_path=None,
        pdb_path_chains=None,
        num_batches=1,
        seed=None,
        chain_id_dict=None,
        fixed_positions_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None,
        backbone_noise=0.0,
        max_length=200000,
    ):
        """
        Calculate sequence unconditional probabilities p(s_i | backbone).

        Args:
            pdb_path_or_str: Path to PDB file OR PDB file content as string
            jsonl_path: Path to JSONL file with parsed structures
            pdb_path_chains: Space-separated chain IDs to design (for PDB input)
            num_batches: Number of batches
            seed: Random seed
            chain_id_dict: Dictionary specifying designed/fixed chains
            fixed_positions_dict: Dictionary with fixed positions
            omit_AA_dict: Per-position amino acids to omit
            tied_positions_dict: Dictionary with tied positions
            pssm_dict: PSSM constraints
            bias_by_res_dict: Per-residue bias
            backbone_noise: Backbone noise level
            max_length: Maximum sequence length

        Returns:
            Dictionary with results for each protein:
            {
                'protein_name': {
                    'log_probs': np.array,  # [num_batches, L, 21]
                    'sequence': np.array,   # [L]
                    'mask': np.array,       # [L]
                    'design_mask': np.array # [L]
                }
            }
        """
        # Set random seed
        if seed is None:
            seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load structure
        dataset, chain_id_dict = self._load_structure(
            pdb_path_or_str=pdb_path_or_str,
            jsonl_path=jsonl_path,
            pdb_path_chains=pdb_path_chains,
            chain_id_dict=chain_id_dict,
            max_length=max_length,
        )

        # Update model's backbone noise
        self.model.augment_eps = backbone_noise

        results = {}

        with torch.no_grad():
            for ix, protein in enumerate(dataset):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]

                # Featurize
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding_all,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    dihedral_mask,
                    tied_pos_list_of_lists_list,
                    pssm_coef,
                    pssm_bias,
                    pssm_log_odds_all,
                    bias_by_res_all,
                    tied_beta,
                ) = tied_featurize(
                    batch_clones,
                    self.device,
                    chain_id_dict,
                    fixed_positions_dict,
                    omit_AA_dict,
                    tied_positions_dict,
                    pssm_dict,
                    bias_by_res_dict,
                    ca_only=self.ca_only,
                )

                name_ = batch_clones[0]["name"]

                log_unconditional_probs_list = []
                for j in range(num_batches):
                    log_unconditional_probs = self.model.unconditional_probs(
                        X, mask, residue_idx, chain_encoding_all
                    )
                    log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())

                concat_log_p = np.concatenate(log_unconditional_probs_list, 0)
                mask_out = (chain_M * chain_M_pos * mask)[0,].cpu().numpy()

                results[name_] = {
                    "log_probs": concat_log_p,
                    "sequence": S[0,].cpu().numpy(),
                    "mask": mask[0,].cpu().numpy(),
                    "design_mask": mask_out,
                }

        return results
