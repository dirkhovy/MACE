#!/usr/bin/env python3
"""
MACE: Multi-Annotator Competence Estimation

MACE is an Expectation-Maximization (EM) algorithm that simultaneously:
- Learns the most likely true labels for items from multiple annotators
- Estimates the competence (reliability) of each annotator

The algorithm models annotators as either "knowing" the correct answer or "guessing"
according to a spamming strategy. It uses EM to iteratively:
1. E-step: Compute expected counts of knowing vs. guessing for each annotator
2. M-step: Update competence estimates and spamming strategies

Features:
- Supports discrete categorical labels (default) and continuous numeric values
- Can incorporate control items (known ground truth) for semi-supervised learning
- Provides confidence estimates via entropy calculations
- Handles missing annotations (empty cells in CSV)

Input Format:
- CSV file with one instance per line
- Each column represents one annotator
- Empty cells indicate missing annotations

Output:
- Predictions: Consensus labels or weighted averages (continuous mode)
- Competence scores: Reliability estimate for each annotator (0-1, higher = more reliable)
- Entropies: Uncertainty measure for each instance (optional)

Original Version: Natural Language Group, April 2013
Current Version: Natural Language Group, April 2013

Copyright (c) 2013 by the University of Southern California
All rights reserved.

Python port of the Java implementation via Cursor, modified to work with Python 3.12, 16 Jan 2026.

Reference:
    Hovy, D., Berg-Kirkpatrick, T., Vaswani, A., & Hovy, E. (2013).
    Learning Whom to Trust With MACE. In: Proceedings of NAACL-HLT.
"""

import sys
import argparse
import time
import numpy as np
from scipy.special import digamma

VERSION = "0.3"

# Defaults
DEFAULT_RR = 10
DEFAULT_ITERATIONS = 50
DEFAULT_NOISE = 0.5
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5


class MACE:
    """
    Multi-Annotator Competence Estimation using EM algorithm.
    
    This class implements the MACE algorithm for aggregating annotations from
    multiple sources while estimating the reliability of each annotator.
    
    Attributes:
        num_instances (int): Number of items/instances to label
        num_annotators (int): Number of annotators
        num_labels (int): Number of unique labels
        spamming (np.ndarray): [num_annotators, 2] probability of guessing vs knowing
        thetas (np.ndarray): [num_annotators, num_labels] spamming strategy distribution
        gold_label_marginals (np.ndarray): [num_instances, num_labels] posterior label probabilities
        log_marginal_likelihood (float): Log likelihood of the data under current model
    
    Example:
        >>> mace = MACE('annotations.csv')
        >>> mace.initialize(0.5)
        >>> mace.run(iterations=50, smoothing=0.01, num_restarts=10, ...)
        >>> predictions = mace.decode(threshold=1.0)
    """
    
    def __init__(self, csv_file, continuous=False):
        """
        Initialize MACE model from CSV file.
        
        Reads annotation data from a CSV file where:
        - Each row represents one instance/item
        - Each column represents one annotator
        - Values are annotations (labels or numeric values)
        - Empty cells indicate missing annotations
        
        Args:
            csv_file (str): Path to CSV file with annotations
            continuous (bool): If True, interpret values as continuous numeric.
                              Returns weighted averages instead of discrete labels.
                              All values must be valid numbers.
        
        Raises:
            IOError: If file cannot be read or has invalid format
            IOError: If continuous=True but non-numeric values found
        
        Example:
            >>> # Discrete labels
            >>> mace = MACE('labels.csv')
            >>> # Continuous values
            >>> mace = MACE('scores.csv', continuous=True)
        """
        self.continuous = continuous
        self.num_instances = self._file_line_count(csv_file)
        
        self.labels = [None] * self.num_instances
        self.who_labeled = [None] * self.num_instances
        # For continuous mode: store original numeric values
        self.continuous_values = [None] * self.num_instances if continuous else None
        
        # Hash stuff
        self.string2int = {}
        self.int2string = []
        self.hash_counter = 0
        
        # Initialize num_annotators before reading file
        self.num_annotators = 0
        
        # Read in CSV file to get all basic information
        self._read_file_data(csv_file)
        
        self.num_labels = len(self.int2string)
        
        if self.num_annotators == 0:
            raise IOError("No annotators found in CSV file!")
        if self.num_labels == 0:
            raise IOError("No labels found in CSV file!")
        
        self.gold_label_marginals = np.zeros((self.num_instances, self.num_labels))
        self.strategy_expected_counts = np.zeros((self.num_annotators, self.num_labels))
        self.knowing_expected_counts = np.zeros((self.num_annotators, 2))
        
        # Parameters
        self.spamming = None
        self.thetas = None
        
        # Priors
        self.theta_priors = None
        self.strategy_priors = None
        
        self.log_marginal_likelihood = 0.0
    
    def initialize(self, init_noise, alpha=None, beta=None):
        """
        Initialize model parameters with random values.
        
        Initializes the spamming probabilities and strategy distributions for each
        annotator. Parameters are randomly initialized with noise to break symmetry
        and then normalized to be valid probability distributions.
        
        Args:
            init_noise (float): Amount of random noise to add (typically 0.5).
                              Higher values create more diverse initializations.
            alpha (float, optional): First hyperparameter for beta prior on knowing
                                    probability. If provided, enables variational inference.
            beta (float, optional): Second hyperparameter for beta prior on knowing
                                   probability. If provided, enables variational inference.
        
        Note:
            Both alpha and beta must be provided together to enable variational mode.
            Otherwise, standard maximum likelihood estimation is used.
        """
        # Vectorized initialization with random noise
        self.spamming = 1.0 + init_noise * np.random.random((self.num_annotators, 2))
        self.thetas = 1.0 + init_noise * np.random.random((self.num_annotators, self.num_labels))
        
        # Normalize rows
        self.spamming /= self.spamming.sum(axis=1, keepdims=True)
        self.thetas /= self.thetas.sum(axis=1, keepdims=True)
        
        if alpha is not None and beta is not None:
            self.theta_priors = np.empty((self.num_annotators, 2))
            self.theta_priors[:, 0] = alpha
            self.theta_priors[:, 1] = beta
            self.strategy_priors = np.full((self.num_annotators, self.num_labels), 10.0)
    
    def e_step(self, controls=None):
        """
        Expectation step: compute expected counts for model parameters.
        
        Computes the posterior probabilities over true labels and expected counts
        of annotator behaviors (knowing vs. guessing) given current parameter estimates.
        This is the E-step of the EM algorithm.
        
        Args:
            controls (dict, optional): Dictionary mapping instance index to known
                                      ground truth label. Used for semi-supervised
                                      learning. Keys are instance indices (0-based),
                                      values are label indices.
        
        Updates:
            - gold_label_marginals: Posterior probability of each label for each instance
            - strategy_expected_counts: Expected counts for each annotator's strategy
            - knowing_expected_counts: Expected counts for knowing vs. guessing
            - log_marginal_likelihood: Log likelihood of data under current model
        """
        # Reset counts
        self.gold_label_marginals.fill(0.0)
        self.knowing_expected_counts.fill(0.0)
        self.strategy_expected_counts.fill(0.0)
        
        # Compute marginals
        self.log_marginal_likelihood = 0.0
        has_controls = controls is not None and len(controls) > 0
        num_labels = self.num_labels
        
        for d in range(self.num_instances):
            labels_d = self.labels[d]
            if labels_d is None:
                continue
                
            who_labeled_d = self.who_labeled[d]
            num_annotators_d = len(labels_d)
            instance_marginal = 0.0
            d_in_controls = has_controls and d in controls
            control_label = controls[d] if d_in_controls else -1
            
            # Compute gold label marginals for each possible label
            for l in range(num_labels):
                gold_label_marginal = 1.0 / num_labels
                
                for ai in range(num_annotators_d):
                    a = who_labeled_d[ai]
                    label_ai = labels_d[ai]
                    prob = self.spamming[a, 0] * self.thetas[a, label_ai]
                    if l == label_ai:
                        prob += self.spamming[a, 1]
                    gold_label_marginal *= prob
                
                # Check controls
                if not d_in_controls or l == control_label:
                    instance_marginal += gold_label_marginal
                    self.gold_label_marginals[d, l] = gold_label_marginal
            
            if instance_marginal > 0:
                self.log_marginal_likelihood += np.log(instance_marginal)
            else:
                self.log_marginal_likelihood = float('-inf')
                continue
            
            inv_instance_marginal = 1.0 / instance_marginal
            
            # Update expected counts
            for ai in range(num_annotators_d):
                a = who_labeled_d[ai]
                label_ai = labels_d[ai]
                spamming_a0 = self.spamming[a, 0]
                spamming_a1 = self.spamming[a, 1]
                theta_a_label = self.thetas[a, label_ai]
                
                if d_in_controls:
                    if label_ai == control_label:
                        l = control_label
                        denom = spamming_a0 * theta_a_label + spamming_a1
                        strategy_marginal = self.gold_label_marginals[d, l] / denom
                        strategy_marginal *= spamming_a0 * theta_a_label
                        norm_strategy = strategy_marginal * inv_instance_marginal
                        self.strategy_expected_counts[a, label_ai] += norm_strategy
                        self.knowing_expected_counts[a, 0] += norm_strategy
                        self.knowing_expected_counts[a, 1] += (
                            self.gold_label_marginals[d, label_ai] * spamming_a1 / 
                            (spamming_a0 * theta_a_label + spamming_a1)
                        ) * inv_instance_marginal
                    else:
                        self.strategy_expected_counts[a, label_ai] += 1.0
                        self.knowing_expected_counts[a, 0] += 1.0
                else:
                    strategy_marginal = 0.0
                    base_denom = spamming_a0 * theta_a_label
                    for l in range(num_labels):
                        denom = base_denom + (spamming_a1 if l == label_ai else 0.0)
                        strategy_marginal += self.gold_label_marginals[d, l] / denom
                    strategy_marginal *= spamming_a0 * theta_a_label
                    norm_strategy = strategy_marginal * inv_instance_marginal
                    self.strategy_expected_counts[a, label_ai] += norm_strategy
                    self.knowing_expected_counts[a, 0] += norm_strategy
                    self.knowing_expected_counts[a, 1] += (
                        self.gold_label_marginals[d, label_ai] * spamming_a1 / 
                        (spamming_a0 * theta_a_label + spamming_a1)
                    ) * inv_instance_marginal
    
    def m_step(self, smoothing):
        """
        Maximization step: update model parameters from expected counts.
        
        Updates the spamming probabilities and strategy distributions based on
        the expected counts computed in the E-step. This is the M-step of the
        EM algorithm using maximum likelihood estimation with smoothing.
        
        Args:
            smoothing (float): Smoothing parameter added to counts before normalization.
                              Prevents zero probabilities. Typically 0.01/num_labels.
        
        Updates:
            - spamming: Probability of knowing vs. guessing for each annotator
            - thetas: Spamming strategy distribution for each annotator
        """
        # Vectorized normalization with smoothing
        smoothed = self.knowing_expected_counts + smoothing
        self.spamming = smoothed / smoothed.sum(axis=1, keepdims=True)
        
        smoothed = self.strategy_expected_counts + smoothing
        self.thetas = smoothed / smoothed.sum(axis=1, keepdims=True)
    
    def variational_m_step(self):
        """
        Variational maximization step: update parameters using Bayesian priors.
        
        Updates model parameters using variational inference with beta priors
        on the knowing probability. Uses the digamma function for proper
        normalization in the variational framework.
        
        Requires:
            - theta_priors: Beta prior parameters for knowing probability
            - strategy_priors: Prior parameters for spamming strategies
        
        Updates:
            - spamming: Variational posterior for knowing vs. guessing
            - thetas: Variational posterior for spamming strategies
        """
        # Vectorized variational normalization
        combined = self.knowing_expected_counts + self.theta_priors
        norm = np.exp(digamma(combined.sum(axis=1, keepdims=True)))
        self.spamming = np.exp(digamma(combined)) / norm
        
        combined = self.strategy_expected_counts + self.strategy_priors
        norm = np.exp(digamma(combined.sum(axis=1, keepdims=True)))
        self.thetas = np.exp(digamma(combined)) / norm
    
    def decode(self, threshold):
        """
        Decode predictions: find most likely label for each instance.
        
        For discrete mode: returns the label with highest posterior probability.
        For continuous mode: returns weighted average of annotator values,
        weighted by their competence scores.
        
        Args:
            threshold (float): Entropy threshold (0.0-1.0) to filter uncertain instances.
                             0.0 = only most certain, 1.0 = all instances.
                             Instances with entropy above threshold return empty string.
        
        Returns:
            list: Predictions for each instance. Empty strings for filtered instances
                  or instances with no annotations.
        
        Example:
            >>> predictions = mace.decode(threshold=0.8)
            >>> # predictions[0] = "cat" (most likely label for instance 0)
        """
        entropies = self.get_label_entropies()
        entropy_threshold = self.get_entropy_for_threshold(threshold)
        
        result = [""] * self.num_instances
        
        if self.continuous:
            # For continuous mode: compute weighted average
            for d in range(self.num_instances):
                if entropies[d] <= entropy_threshold and entropies[d] != float('-inf'):
                    if self.continuous_values[d] is not None and len(self.continuous_values[d]) > 0:
                        # Get annotators and their values for this instance
                        annotators = self.who_labeled[d]
                        values = self.continuous_values[d]
                        # Get competence scores (spamming[a, 1] = probability of knowing)
                        competences = self.spamming[annotators, 1]
                        # Compute weighted average
                        weighted_sum = np.sum(values * competences)
                        total_weight = np.sum(competences)
                        if total_weight > 0:
                            result[d] = str(weighted_sum / total_weight)
                        else:
                            result[d] = str(np.mean(values))  # Fallback to simple mean
        else:
            # Discrete mode: original behavior
            for d in range(self.num_instances):
                if entropies[d] <= entropy_threshold and entropies[d] != float('-inf'):
                    best_label = np.argmax(self.gold_label_marginals[d])
                    result[d] = self.int2string[best_label]
        
        return result
    
    def decode_distribution(self, threshold):
        """
        Decode predictions with full distribution information.
        
        Returns full probability distributions over labels or statistical summaries
        for continuous values, rather than just the most likely prediction.
        
        Args:
            threshold (float): Entropy threshold (0.0-1.0) to filter uncertain instances.
        
        Returns:
            list: For each instance, returns:
                - Discrete mode: Tab-separated "label probability" pairs, sorted by
                  probability descending (e.g., "cat 0.8\tdog 0.15\tbird 0.05")
                - Continuous mode: Tab-separated stats "mean\tstd\tmin\tmax\tn_annotators"
                - Empty strings for filtered instances
        
        Example:
            >>> dists = mace.decode_distribution(threshold=1.0)
            >>> # dists[0] = "cat 0.8\tdog 0.15\tbird 0.05"
        """
        entropies = self.get_label_entropies()
        entropy_threshold = self.get_entropy_for_threshold(threshold)
        
        result = [""] * self.num_instances
        
        if self.continuous:
            # For continuous mode: return weighted mean, std, and individual values with weights
            for d in range(self.num_instances):
                if entropies[d] <= entropy_threshold and entropies[d] != float('-inf'):
                    if self.continuous_values[d] is not None and len(self.continuous_values[d]) > 0:
                        annotators = self.who_labeled[d]
                        values = self.continuous_values[d]
                        competences = self.spamming[annotators, 1]
                        
                        # Weighted mean
                        weighted_sum = np.sum(values * competences)
                        total_weight = np.sum(competences)
                        if total_weight > 0:
                            weighted_mean = weighted_sum / total_weight
                            # Weighted variance
                            weighted_variance = np.sum(competences * (values - weighted_mean)**2) / total_weight
                            weighted_std = np.sqrt(weighted_variance) if weighted_variance > 0 else 0.0
                        else:
                            weighted_mean = np.mean(values)
                            weighted_std = np.std(values)
                        
                        # Format: mean, std, min, max, n_annotators
                        result[d] = f"{weighted_mean}\t{weighted_std}\t{np.min(values)}\t{np.max(values)}\t{len(values)}"
        else:
            # Discrete mode: original behavior
            for d in range(self.num_instances):
                if entropies[d] <= entropy_threshold and entropies[d] != float('-inf'):
                    marginals = self.gold_label_marginals[d]
                    # Sort indices by value descending
                    sorted_indices = np.argsort(marginals)[::-1]
                    output = [f"{self.int2string[i]} {marginals[i]}" for i in sorted_indices]
                    result[d] = "\t".join(output)
        
        return result
    
    def get_label_entropies(self):
        """
        Compute entropy of label distribution for each instance.
        
        Entropy measures uncertainty in the predicted label distribution.
        Higher entropy indicates more uncertainty (annotators disagree).
        Lower entropy indicates high confidence (annotators agree).
        
        Returns:
            np.ndarray: Entropy values for each instance.
                      Returns -inf for instances with no annotations.
        
        Example:
            >>> entropies = mace.get_label_entropies()
            >>> # entropies[0] = 0.45  (moderate uncertainty)
            >>> # entropies[1] = 0.12  (high confidence)
        """
        result = np.full(self.num_instances, float('-inf'))
        
        for d in range(self.num_instances):
            if self.labels[d] is not None:
                marginals = self.gold_label_marginals[d]
                norm = marginals.sum()
                if norm > 0:
                    p = marginals / norm
                    # Avoid log(0) by masking zeros
                    mask = p > 0
                    result[d] = -np.sum(p[mask] * np.log(p[mask]))
        
        return result
    
    def run(self, num_iters, smoothing, num_restarts, alpha, beta, variational, controls_file):
        """
        Run the EM algorithm to learn annotator competences and true labels.
        
        Performs multiple random restarts of the EM algorithm and selects the model
        with highest log marginal likelihood. Each restart:
        1. Randomly initializes parameters
        2. Alternates E-step and M-step for specified iterations
        3. Tracks the best model across all restarts
        
        Args:
            num_iters (int): Number of EM iterations per restart (typically 50)
            smoothing (float): Smoothing parameter for M-step (typically 0.01/num_labels)
            num_restarts (int): Number of random restarts (typically 10)
            alpha (float): First hyperparameter for beta prior (variational mode)
            beta (float): Second hyperparameter for beta prior (variational mode)
            variational (bool): If True, use variational inference instead of MLE
            controls_file (str, optional): Path to file with control items (known labels)
        
        Updates:
            - spamming: Best competence estimates found across all restarts
            - thetas: Best strategy distributions found across all restarts
            - gold_label_marginals: Posterior probabilities for best model
            - log_marginal_likelihood: Log likelihood of best model
        
        Note:
            Prints progress information to stderr including which restart produced
            the best model.
        """
        controls = self._read_controls(controls_file) if controls_file else {}
        
        best_thetas = None
        best_strategies = None
        best_log_marginal_likelihood = float('-inf')
        rr_best_model_occurred_at = 0
        
        print("Running training with the following settings:", file=sys.stderr)
        print(f"\t{num_iters} iterations", file=sys.stderr)
        print(f"\t{num_restarts} restarts", file=sys.stderr)
        print(f"\tsmoothing = {smoothing}", file=sys.stderr)
        if variational:
            print(f"\talpha = {alpha}", file=sys.stderr)
            print(f"\tbeta = {beta}", file=sys.stderr)
        
        start = time.time()
        for rr in range(num_restarts):
            print(f"\n============\nRestart {rr + 1}\n============", file=sys.stderr)
            
            # Initialize
            if variational:
                self.initialize(DEFAULT_NOISE, alpha, beta)
            else:
                self.initialize(DEFAULT_NOISE)
            
            # Run first E-Step to get counts
            self.e_step(controls)
            print(f"initial log marginal likelihood = {self.log_marginal_likelihood}", file=sys.stderr)
            
            # Iterate
            for _ in range(num_iters):
                if variational:
                    self.variational_m_step()
                else:
                    self.m_step(smoothing)
                self.e_step(controls)
            
            print(f"final log marginal likelihood = {self.log_marginal_likelihood}", file=sys.stderr)
            
            if self.log_marginal_likelihood > best_log_marginal_likelihood:
                rr_best_model_occurred_at = rr + 1
                best_log_marginal_likelihood = self.log_marginal_likelihood
                best_thetas = self.spamming.copy()
                best_strategies = self.thetas.copy()
        
        elapsed = time.time() - start
        print(f"\nTraining completed in {elapsed:.2f}sec", file=sys.stderr)
        print(f"Best model came from random restart number {rr_best_model_occurred_at} "
              f"(log marginal likelihood: {best_log_marginal_likelihood})", file=sys.stderr)
        self.log_marginal_likelihood = best_log_marginal_likelihood
        self.spamming = best_thetas
        self.thetas = best_strategies
        
        # Run E-step to get marginals of latest model
        self.e_step(controls)
    
    def _read_controls(self, file_name):
        """
        Read control items (known ground truth labels) from file.
        
        Control items are used for semi-supervised learning. They provide
        known labels for specific instances, which helps guide the EM algorithm.
        
        Args:
            file_name (str): Path to file with control labels, one per line.
                           Empty lines indicate no control for that instance.
                           Must have same number of lines as input CSV.
        
        Returns:
            dict: Mapping from instance index (0-based) to label index.
                 Only includes entries for non-empty lines.
        
        Raises:
            IOError: If file cannot be read
        """
        controls = {}
        
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                print(f"Reading controls file {file_name}", file=sys.stderr)
                
                for line_number, line in enumerate(f):
                    line = line.strip()
                    if line:
                        if line not in self.string2int:
                            self.string2int[line] = self.hash_counter
                            self.int2string.append(line)
                            self.hash_counter += 1
                        controls[line_number] = self.string2int[line]
            
            return controls
        
        except Exception as e:
            raise IOError(f"Problem reading file: {str(e)}")
    
    def _read_file_data(self, file_name):
        """
        Read annotation data from CSV file.
        
        Parses CSV file where each row is an instance and each column is an annotator.
        Empty cells indicate missing annotations. In continuous mode, validates
        that all values are numeric.
        
        Args:
            file_name (str): Path to CSV file with annotations
        
        Raises:
            IOError: If file cannot be read
            IOError: If continuous mode but non-numeric values found
            IOError: If number of columns varies between rows
        
        Updates:
            - labels: Label indices for each instance
            - who_labeled: Annotator indices for each annotation
            - continuous_values: Original numeric values (if continuous mode)
            - string2int, int2string: Mappings between labels and indices
            - num_annotators: Number of annotators (columns)
        """
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                print(f"Reading CSV file '{file_name}'", file=sys.stderr)
                
                for line_number, line in enumerate(f):
                    if line_number > 0 and line_number % 5 == 0:
                        print(".", end="", file=sys.stderr, flush=True)
                    if line_number > 0 and line_number % 100 == 0:
                        print(f"\n{line_number}", file=sys.stderr)
                    
                    # Handle empty lines
                    if not line.strip():
                        self.labels[line_number] = None
                        self.who_labeled[line_number] = None
                        if self.continuous:
                            self.continuous_values[line_number] = None
                        continue
                    
                    tokens = line.rstrip('\n').split(',')
                    num_annotators_this_line = len(tokens)
                    
                    if self.num_annotators > 0 and num_annotators_this_line != self.num_annotators:
                        raise IOError(f"number of annotations in line {line_number + 1} differs from previous line!")
                    self.num_annotators = num_annotators_this_line
                    
                    annotators_on_item = []
                    item_values = []
                    continuous_vals = [] if self.continuous else None
                    
                    for annotator_number, token in enumerate(tokens):
                        item = token.strip()
                        if item:
                            annotators_on_item.append(annotator_number)
                            
                            if self.continuous:
                                # Validate that it's a number
                                try:
                                    numeric_value = float(item)
                                    continuous_vals.append(numeric_value)
                                except ValueError:
                                    raise IOError(f"Line {line_number + 1}, column {annotator_number + 1}: '{item}' is not a valid number (--continuous mode requires numeric values)")
                                
                                # Still use string representation for discrete model
                                item_str = item
                            else:
                                item_str = item
                            
                            if item_str not in self.string2int:
                                self.string2int[item_str] = self.hash_counter
                                self.int2string.append(item_str)
                                self.hash_counter += 1
                            item_values.append(self.string2int[item_str])
                    
                    # Store as arrays for faster access
                    self.labels[line_number] = np.array(item_values, dtype=np.int32) if item_values else None
                    self.who_labeled[line_number] = np.array(annotators_on_item, dtype=np.int32) if annotators_on_item else None
                    if self.continuous:
                        self.continuous_values[line_number] = np.array(continuous_vals, dtype=np.float64) if continuous_vals else None
            
            print(f"\nstats:\n\t{self.num_instances} instances,\n\t{len(self.int2string)} labels {self.int2string},\n\t{self.num_annotators} annotators\n", file=sys.stderr)
            if self.continuous:
                print(f"\tMode: Continuous (numeric values)", file=sys.stderr)
        
        except Exception as e:
            raise IOError(f"Problem reading file: {str(e)}")
    
    def _file_line_count(self, filename):
        """
        Count the number of lines in a file.
        
        Used to pre-allocate arrays before reading the file.
        
        Args:
            filename (str): Path to file to count lines in
        
        Returns:
            int: Number of lines in the file
        
        Raises:
            IOError: If file cannot be read
        """
        try:
            with open(filename, 'rb') as f:
                return sum(1 for _ in f)
        except Exception as e:
            raise IOError(f"Problem reading file: {str(e)}")
    
    def write_array_to_file(self, array, file_name, delimiter, header=None):
        """
        Write an array to a file with optional header.
        
        Writes array elements separated by the specified delimiter. If header
        is provided, writes it as the first line.
        
        Args:
            array (list): Array of values to write
            file_name (str): Output file path
            delimiter (str): String to separate elements (e.g., "\\n", "\\t", ",")
            header (str, optional): Header line to write before data
        
        Raises:
            IOError: If file cannot be written
        
        Example:
            >>> mace.write_array_to_file([1, 2, 3], "out.txt", "\\n", header="values")
            >>> # Writes: "values\\n1\\n2\\n3\\n"
        """
        try:
            print(f"writing to file '{file_name}'...", end="", file=sys.stderr)
            with open(file_name, 'w', encoding='utf-8') as f:
                if header:
                    f.write(header + "\n")
                f.write(delimiter.join(str(item) for item in array))
                f.write("\n")
            print("done", file=sys.stderr)
        except Exception as e:
            raise IOError(f"Problem writing file: {str(e)}")
    
    def get_entropy_for_threshold(self, threshold):
        """
        Get entropy value corresponding to threshold percentile.
        
        Sorts all entropy values and returns the value at the specified percentile.
        Used to filter instances: only instances with entropy <= this value are
        included in predictions.
        
        Args:
            threshold (float): Percentile threshold (0.0-1.0).
                             0.0 = minimum entropy (most certain)
                             1.0 = maximum entropy (all instances)
        
        Returns:
            float: Entropy value at the specified percentile
        
        Example:
            >>> # Get entropy threshold for top 80% most certain instances
            >>> threshold_val = mace.get_entropy_for_threshold(0.8)
        """
        if threshold == 0.0:
            pivot = 0
        elif threshold == 1.0:
            pivot = self.num_instances - 1
        else:
            pivot = int(self.num_instances * threshold)
        
        entropy_array = self.get_label_entropies()
        return np.partition(entropy_array, pivot)[pivot]
    
    def test(self, test_file, predictions, distribution_format=False):
        """
        Evaluate model predictions against gold standard labels.
        
        Compares predictions to ground truth labels from a test file and computes
        evaluation metrics. Only evaluates instances that have predictions (not
        filtered by threshold).
        
        Args:
            test_file (str): Path to file with gold standard labels, one per line.
                           Must have same number of lines as input CSV.
            predictions (list): Model predictions from decode() or decode_distribution()
            distribution_format (bool): If True, predictions are in distribution format
                                      (tab-separated). Extracts best prediction automatically.
        
        Returns:
            float: Evaluation metric:
                  - Discrete mode: Accuracy (proportion of correct predictions)
                  - Continuous mode: RMSE (Root Mean Squared Error)
        
        Raises:
            IOError: If test file cannot be read or has wrong number of lines
            IOError: If continuous mode but non-numeric values in test file
        
        Note:
            Prints coverage (proportion of instances with predictions) to stdout.
        """
        num_lines_in_test = self._file_line_count(test_file)
        if num_lines_in_test != self.num_instances:
            raise IOError("Number of lines in test file does not match!")
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                print("Reading test file", file=sys.stderr)
                
                if self.continuous:
                    # Continuous mode: compute RMSE
                    squared_errors = []
                    total = 0
                    
                    for line_number, line in enumerate(f):
                        if predictions[line_number]:
                            total += 1
                            try:
                                actual = float(line.strip())
                                pred_str = predictions[line_number]
                                
                                # If distribution format, extract first value (weighted mean)
                                if distribution_format:
                                    pred_str = pred_str.split('\t')[0]
                                
                                predicted = float(pred_str)
                                squared_errors.append((predicted - actual) ** 2)
                            except (ValueError, IndexError) as e:
                                raise IOError(f"Line {line_number + 1}: could not parse prediction '{predictions[line_number]}' (continuous mode requires numeric values)")
                    
                    coverage = total / self.num_instances
                    print(f"Coverage: {coverage}\t", end="")
                    
                    if total > 0 and len(squared_errors) > 0:
                        mse = np.mean(squared_errors)
                        rmse = np.sqrt(mse)
                        return rmse
                    else:
                        return float('inf')
                else:
                    # Discrete mode: compute accuracy
                    correct = 0
                    total = 0
                    
                    for line_number, line in enumerate(f):
                        if predictions[line_number]:
                            total += 1
                            pred_str = predictions[line_number]
                            actual_str = line.strip()
                            
                            # If distribution format, extract first label (highest probability)
                            if distribution_format:
                                # Format is "label prob\tlabel prob\t..."
                                first_pair = pred_str.split('\t')[0]
                                # Extract label (everything before the last space)
                                pred_str = first_pair.rsplit(' ', 1)[0]
                            
                            if actual_str == pred_str:
                                correct += 1
                    
                    coverage = total / self.num_instances
                    print(f"Coverage: {coverage}\t", end="")
                    return correct / total if total > 0 else 0.0
        
        except Exception as e:
            raise IOError(f"Problem reading test file: {str(e)}")


def main():
    """
    Main entry point for MACE command-line interface.
    
    Parses command-line arguments, runs the MACE algorithm, and writes results
    to output files. Supports both discrete categorical labels and continuous
    numeric values.
    
    Output Files:
        - {prefix}.prediction: Consensus predictions (or distributions if --distribution)
        - {prefix}.competence: Competence scores for each annotator
        - {prefix}.entropies: Entropy values for each instance (if --entropies)
    
    Exit Codes:
        0: Success
        1: File error or argument error
    
    Example:
        >>> # Basic usage with discrete labels
        >>> python mace.py example.csv
        >>> 
        >>> # Continuous values with test evaluation
        >>> python mace.py --continuous --test gold.txt scores.csv
        >>> 
        >>> # With control items and custom prefix
        >>> python mace.py --controls controls.txt --prefix results example.csv
    """
    parser = argparse.ArgumentParser(
        description="MACE: Multi-Annotator Competence Estimation - EM algorithm for learning true labels and annotator reliability from multiple annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage with discrete labels:
    %(prog)s annotations.csv
  
  Continuous numeric values with test evaluation:
    %(prog)s --continuous --test gold_standard.txt scores.csv
  
  With control items and custom output prefix:
    %(prog)s --controls known_labels.txt --prefix results annotations.csv
  
  Distribution output with headers:
    %(prog)s --distribution --headers --prefix output annotations.csv

Input Format:
  CSV file with one instance per line, each column is one annotator.
  Empty cells indicate missing annotations. In continuous mode, all values must be numeric.

Output Files:
  {prefix}.prediction    - Consensus predictions (or distributions if --distribution)
  {prefix}.competence    - Competence scores for each annotator (0-1, higher = more reliable)
  {prefix}.entropies     - Uncertainty measure for each instance (if --entropies)

Citation:
  Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani and Eduard Hovy (2013): 
  Learning Whom to Trust With MACE. In: Proceedings of NAACL-HLT. 
  Association for Computational Linguistics.

This is research software that is not actively maintained.
        """
    )
    
    parser.add_argument('--version', action='version', version=f'MACE {VERSION}')
    parser.add_argument('--controls', type=str, metavar='FILE',
                       help='File with control items (known ground truth labels) for semi-supervised learning. '
                            'One label per line, empty lines indicate no control. Must match number of instances.')
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, metavar='FLOAT',
                       help=f'First hyperparameter of beta prior for variational inference. '
                            f'Enables variational mode when set. Default: {DEFAULT_ALPHA}')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA, metavar='FLOAT',
                       help=f'Second hyperparameter of beta prior for variational inference. '
                            f'Enables variational mode when set. Default: {DEFAULT_BETA}')
    parser.add_argument('--continuous', action='store_true',
                       help='Interpret input values as continuous numeric. Returns weighted averages '
                            'weighted by annotator competence scores. All values must be valid numbers. '
                            'Test evaluation uses RMSE instead of accuracy.')
    parser.add_argument('--distribution', action='store_true',
                       help='Output full probability distributions instead of single predictions. '
                            'Discrete: tab-separated "label probability" pairs sorted by probability. '
                            'Continuous: tab-separated stats "mean\\tstd\\tmin\\tmax\\tn_annotators".')
    parser.add_argument('--entropies', action='store_true',
                       help='Write entropy values (uncertainty measure) for each instance to a separate file. '
                            'Higher entropy indicates more disagreement among annotators.')
    parser.add_argument('--headers', action='store_true',
                       help='Add header rows to output files describing column contents.')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS, metavar='N',
                       help=f'Number of EM iterations per random restart. More iterations may improve '
                            f'convergence but increase runtime. Default: {DEFAULT_ITERATIONS}')
    parser.add_argument('--prefix', type=str, metavar='PREFIX',
                       help='Prefix for output filenames. If not specified, uses default names '
                            '(prediction, competence, entropies). Output: {prefix}.prediction, etc.')
    parser.add_argument('--restarts', type=int, default=DEFAULT_RR, metavar='N',
                       help=f'Number of random restarts to perform. Multiple restarts help avoid '
                            f'local optima. Best model (highest likelihood) is selected. Default: {DEFAULT_RR}')
    parser.add_argument('--smoothing', type=float, metavar='FLOAT',
                       help='Smoothing parameter added to expected counts before normalization. '
                            'Prevents zero probabilities. If not specified, defaults to 0.01/num_labels. '
                            'Higher values = more conservative updates.')
    parser.add_argument('--test', type=str, metavar='FILE',
                       help='Test file with gold standard labels for evaluation. One label per line, '
                            'must match number of instances. Reports accuracy (discrete) or RMSE (continuous).')
    parser.add_argument('--threshold', type=float, default=1.0, metavar='FLOAT',
                       help='Entropy threshold (0.0-1.0) to filter uncertain instances. '
                            '0.0 = only most certain, 1.0 = all instances. '
                            'Instances above threshold are not predicted (empty in output). Default: 1.0')
    parser.add_argument('csv_file', metavar='CSV_FILE',
                       help='Input CSV file with annotations. Each row is an instance, each column is an annotator.')
    
    args = parser.parse_args()
    
    try:
        em = MACE(args.csv_file, continuous=args.continuous)
        
        smoothing = args.smoothing if args.smoothing is not None else 0.01 / em.num_labels
        variational = args.alpha != DEFAULT_ALPHA or args.beta != DEFAULT_BETA
        
        # Validate arguments
        if smoothing < 0.0:
            raise ValueError("smoothing less than 0.0")
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("threshold not between 0.0 and 1.0")
        if not 1 <= args.restarts <= 1000:
            raise ValueError("restarts not between 1 and 1000")
        if not 1 <= args.iterations <= 1000:
            raise ValueError("iterations not between 1 and 1000")
        
        # Run with configuration
        em.run(args.iterations, smoothing, args.restarts, args.alpha, args.beta, variational, args.controls)
        
        # Write results to files
        predictions = em.decode_distribution(args.threshold) if args.distribution else em.decode(args.threshold)
        
        prefix = args.prefix
        prediction_name = f"{prefix}.prediction" if prefix else "prediction"
        
        # Generate header for prediction file
        pred_header = None
        if args.headers:
            if args.distribution:
                if args.continuous:
                    pred_header = "weighted_mean\tweighted_std\tmin\tmax\tn_annotators"
                else:
                    # For discrete distribution, create header with actual label names
                    # Note: output is sorted by probability, but header shows all labels
                    header_parts = []
                    for i in range(em.num_labels):
                        header_parts.append(f"{em.int2string[i]}\tprob_{em.int2string[i]}")
                    pred_header = "\t".join(header_parts)
            else:
                pred_header = "prediction"
        
        em.write_array_to_file(predictions, prediction_name, "\n", header=pred_header)
        
        # Generate competence scores
        competence = em.spamming[:, 1].tolist()
        competence_name = f"{prefix}.competence" if prefix else "competence"
        
        # Generate header for competence file
        comp_header = None
        if args.headers:
            # Tab-separated annotator IDs
            comp_header = "\t".join(f"annotator_{i}" for i in range(em.num_annotators))
        
        em.write_array_to_file(competence, competence_name, "\t", header=comp_header)
        
        # Generate entropies
        if args.entropies:
            entropy = em.get_label_entropies().tolist()
            entropy_name = f"{prefix}.entropies" if prefix else "entropies"
            
            # Generate header for entropy file
            entropy_header = "entropy" if args.headers else None
            
            em.write_array_to_file(entropy, entropy_name, "\n", header=entropy_header)
        
        if args.test:
            result = em.test(args.test, predictions, distribution_format=args.distribution)
            metric_name = "RMSE" if args.continuous else "Accuracy"
            print(f"{metric_name} on test set: {result}")
    
    except IOError as e:
        print("\n*****************************************************************", file=sys.stderr)
        print("\tFILE ERROR:", file=sys.stderr)
        print(f"\t{str(e)}", file=sys.stderr)
        print("*****************************************************************", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print("\n*****************************************************************", file=sys.stderr)
        print("\tARGUMENT ERROR:", file=sys.stderr)
        print(f"\t{str(e)}", file=sys.stderr)
        print("*****************************************************************", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
