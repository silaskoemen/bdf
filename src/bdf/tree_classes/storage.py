"""General storage for functions that have changed since for legacy reasons.
Includes the `heapq` fit method for the tree under dynamically growing penalties.
"""

# Legacy function for dynamic penalties
# def fit(
#     self,
#     X: np.ndarray,
#     y: np.ndarray,
#     col_idcs: list | np.ndarray | None = None,
#     verbose: bool = False,
#     eta: float = 0.025,
# ) -> "BDFTree":
#     # NOTE: if reg_lambda is 0, each loss component is fully seperable, meaning each
#     # node can be split simply by considering NLL reduction and including reg_beta; no need for a queue.
#     # TODO: Implement separate fit functions given reg_lambda == 0 and reg_lambda > 0.
#     # Create root node
#     self.root = BDFNode(distribution=self.distribution, depth=0, random_state=self.random_state)
#     self.root.estimate_posterior(y)
#     self.n_leaves = 1

#     # Calculate initial tree loss (NLL at root + regularization)
#     root_nll = self.distribution.nll(y)
#     self.tree_loss = root_nll + self.reg_beta + self.reg_lambda

#     # Initialize priority queue with potential splits
#     split_candidates: list[tuple] = []

#     # Only try splitting if we have enough samples
#     if len(y) >= self.min_samples_split:
#         # Evaluate initial split of the root
#         feature_idx, threshold, loss_reduction, left_indices, right_indices, left_params, right_params = (
#             self.root.find_best_split(
#                 X, y, self.min_samples_leaf, self.min_child_weight, col_idcs=col_idcs, eta=eta
#             )
#         )

#         if loss_reduction > 0 and feature_idx is not None:
#             heapq.heappush(
#                 split_candidates,
#                 (
#                     -loss_reduction,
#                     id(self.root),
#                     self.root,
#                     X,
#                     y,
#                     feature_idx,
#                     threshold,
#                     left_indices,
#                     right_indices,
#                     left_params,
#                     right_params,
#                 ),
#             )

#     # Process queue until no more beneficial splits
#     while split_candidates:
#         # Get best split from queue
#         neg_gain, _, node, node_X, node_y, feat_idx, thresh, left_idx, right_idx, left_params, right_params = (
#             heapq.heappop(split_candidates)
#         )
#         nll_reduction = -neg_gain
#         # Check whether split given tree structure and regularization is still beneficial
#         # Calculate exact penalty for adding one leaf node
#         # When splitting, one leaf becomes two leaves (net +1)
#         # Old penalty: reg_beta * n_leaves + reg_lambda * n_leaves^2
#         # New penalty: reg_beta * (n_leaves+1) + reg_lambda * (n_leaves+1)^2
#         # Penalty increase: reg_beta + reg_lambda * (2*n_leaves + 1)
#         if self.reg_beta + self.reg_lambda * (2 * self.n_leaves + 1) - nll_reduction <= 0:
#             # Execute the split
#             node.split_node(y=node_y, feat_idx=feat_idx, threshold=thresh, left_idx=left_idx, right_idx=right_idx)

#             # Get the data for each child
#             X_left, y_left = node_X[left_idx], node_y[left_idx]
#             X_right, y_right = node_X[right_idx], node_y[right_idx]

#             # Update global tree metrics
#             self.n_leaves += 1  # One leaf becomes two, net +1

#             # Update tree loss
#             self.tree_loss -= nll_reduction + self.reg_beta + self.reg_lambda * (2 * (self.n_leaves - 1) + 1)

#             # Evaluate further splits for each new child
#             for child_node, child_X, child_y in [
#                 (node.left_node, X_left, y_left),
#                 (node.right_node, X_right, y_right),
#             ]:
#                 # Only try splitting if we have enough samples
#                 if len(child_y) >= self.min_samples_split and child_node.depth < self.max_depth:
#                     # Find best split for this child
#                     c_feat, c_thresh, c_loss_reduction, c_left_idx, c_right_idx, c_left_params, c_right_params = child_node.find_best_split(
#                         child_X, child_y, self.min_samples_leaf, self.min_child_weight, col_idcs=col_idcs, eta=eta
#                     )

#                     # Only add to queue if split is beneficial
#                     if c_loss_reduction > 0 and c_feat is not None:
#                         heapq.heappush(
#                             split_candidates,
#                             (
#                                 -c_loss_reduction,
#                                 id(child_node),
#                                 child_node,
#                                 child_X,
#                                 child_y,
#                                 c_feat,
#                                 c_thresh,
#                                 c_left_idx,
#                                 c_right_idx,
#                                 c_left_params,
#                                 c_right_params,
#                             ),
#                         )
#         else:
#             break  # No more beneficial splits, exit loop
#     if verbose:
#         print(f"Tree built with {self.n_leaves} leaves and total loss: {self.tree_loss:.4f}")
#     return self
